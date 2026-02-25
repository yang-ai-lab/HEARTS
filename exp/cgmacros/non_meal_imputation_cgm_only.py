import uuid
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List
import json_repair
import numpy as np
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics

@register_experiment("non_meal_imputation_cgm_only")
class NonMealImputationCGMOnlyExperiment(CGMacrosExperiment):
    """
    Experiment for imputing CGM values in a non-meal window using heart rate and remaining CGM data.
    The agent is given a 2-hour CGM + HR window with no meal events. A random 30-min CGM segment is masked (set to 0), and the agent must impute it using heart rate and the rest of CGM.
    """
    DEFAULT_WINDOW_HOURS = 2
    DEFAULT_MASK_MINUTES = 30
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"
    PRED_KEY = "imputed_cgm"

    def __init__(
        self,
        task: str = "non_meal_imputation_cgm_only",
        num_test: int = 50,
        logs_dir=None,
        agent=None,
        data_dir=None,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        mask_minutes: int = DEFAULT_MASK_MINUTES,
    ):
        super().__init__(task, num_test, logs_dir, agent, data_dir)
        self.window_hours = window_hours
        self.mask_minutes = mask_minutes

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = str(uuid.uuid4())
        file_name = "cgm_only"

        mask_start = data["mask_start"]
        mask_end = data["mask_end"]

        agent_input_df = data["window_df"][[self.TIMESTAMP_COLUMN, self.CGM_COLUMN]]
        agent_input_data = {f"{file_name}": agent_input_df}

        self.save_data(agent_input_data, query_id=query_id)

        prompt = dedent(f"""\
        The CGM (continuous glucose monitoring) data for this subject is provided in 'input/{file_name}.csv'. There are two columns: Timestamp (datetime) and 'Libre GL' (CGM, mg/dL).
        In this {self.window_hours}-hour window, a {self.mask_minutes}-min segment of CGM readings is corrupted (set to 0), start from {mask_start} and end at {mask_end}. Please impute the missing CGM values in this segment using the remaining CGM data. Then save your imputed values in order as a list in json file at `output/solution.json`. When finished, put your methodology to solve this task as your answer.
        """)
        _ = await self.agent.query(
            prompt,
            agent_input_data,
            self.logs_dir,
            query_id=query_id,
        )

        # parse output file here
        try:
            with open(
                Path("agent_working") / query_id / "output" / "solution.json", "r"
            ) as f:
                solution = json_repair.load(f)
            assert isinstance(solution, list), f"Solution is not a list: {solution}"
            assert len(solution) == len(data["GT"]), (
                f"Solution length {len(solution)} mismatch GT length {len(data['GT'])}. Content of solution: {solution}"
            )
        except Exception as e:
            solution = {"Error": f"Error loading solution.json: {e}"}

        result = {
            "query_id": query_id,
            "subject_id": data["subject_id"],
            "mask_start": data["mask_start"],
            "mask_end": data["mask_end"],
            "GT": data["GT"],
            "solution": solution,
        }
        return result

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        if not result_list:
            return {"MAE": 0.0, "MSE": 0.0, "MAPE": 0.0, "Failures": 0}
        preds = defaultdict(list)
        gts = defaultdict(list)
        failure_count = 0
        smape_invalid_count = 0
        minmax_smape_invalid_count = 0  # for minmax SMAPE
        for result in result_list:
            pred = result["solution"]
            gt = result["GT"]
            if pred and isinstance(pred, list) and len(pred) == len(gt):
                preds[self.PRED_KEY].extend(pred)
                gts[self.PRED_KEY].extend(gt)
            else:
                failure_count += 1
        mae = (
            Metrics.mae(preds, gts)[self.PRED_KEY]
            if preds[self.PRED_KEY]
            else float("nan")
        )
        mse = (
            Metrics.mse(preds, gts)[self.PRED_KEY]
            if preds[self.PRED_KEY]
            else float("nan")
        )
        mape = (
            Metrics.mape(preds, gts)[self.PRED_KEY]
            if preds[self.PRED_KEY]
            else float("nan")
        )
        smape_dict, invalid_count = (
            Metrics.smape(preds, gts) if preds[self.PRED_KEY] else ({}, 0)
        )
        smape = smape_dict.get(self.PRED_KEY, float("nan"))
        smape_invalid_count += invalid_count

        # Calculate MINMAX_SMAPE
        minmax_smape = float("nan")
        if preds[self.PRED_KEY]:
            pred_array = np.array(preds[self.PRED_KEY])
            gt_array = np.array(gts[self.PRED_KEY])

            # Apply minmax scaling using common min and max
            pred_min = np.min(pred_array)
            pred_max = np.max(pred_array)
            gt_min = np.min(gt_array)
            gt_max = np.max(gt_array)

            common_min = min(pred_min, gt_min)
            common_max = max(pred_max, gt_max)

            # Avoid division by zero
            if common_max - common_min > 0:
                pred_array_scaled = (pred_array - common_min) / (
                    common_max - common_min
                )
                gt_array_scaled = (gt_array - common_min) / (common_max - common_min)
            else:
                pred_array_scaled = pred_array
                gt_array_scaled = gt_array

            # Calculate SMAPE on scaled arrays
            if np.all(gt_array_scaled == 0):
                minmax_smape_result, minmax_invalid_count = Metrics.smape(
                    {self.PRED_KEY: pred_array}, {self.PRED_KEY: gt_array}
                )
            else:
                minmax_smape_result, minmax_invalid_count = Metrics.smape(
                    {self.PRED_KEY: pred_array_scaled.tolist()},
                    {self.PRED_KEY: gt_array_scaled.tolist()},
                )
            minmax_smape = minmax_smape_result.get(self.PRED_KEY, float("nan"))
            minmax_smape_invalid_count += minmax_invalid_count

        # Calculate overall SMAPE across all samples
        print("Failures:", failure_count)
        print("MAE:", round(mae, 4))
        print("MSE:", round(mse, 4))
        print("MAPE:", round(mape, 4))
        print("SMAPE:", round(smape, 4))
        print("MINMAX_SMAPE:", round(minmax_smape, 4))
        return {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "MAPE": round(mape, 4),
            "SMAPE": round(smape, 4),
            "MINMAX_SMAPE": round(minmax_smape, 4),
            "Failures": failure_count,
            "additional_metrics": {
                "smape_invalid_count": smape_invalid_count,
                "minmax_smape_invalid_count": minmax_smape_invalid_count,
            },
        }

    def parse_output(self, content):
        pass
