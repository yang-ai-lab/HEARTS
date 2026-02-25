import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List
import json_repair
import numpy as np
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics

@register_experiment("non_meal_imputation_calories")
class NonMealImputationCaloriesExperiment(CGMacrosExperiment):
    """
    Experiment for imputing CGM values in a non-meal window using calories and remaining CGM data.
    The agent is given a 2-hour CGM + Calories window with no meal events. A random 30-min CGM segment is masked (set to 0), and the agent must impute it using calories and the rest of CGM.
    """
    # Default values, now set via __init__
    DEFAULT_WINDOW_HOURS = 2
    DEFAULT_MASK_MINUTES = 30
    CGM_COLUMN = "Libre GL"
    CAL_COLUMN = "Calories (Activity)"
    TIMESTAMP_COLUMN = "Timestamp"
    PRED_KEY = "imputed_cgm"

    def __init__(
        self,
        task: str = "non_meal_imputation_calories",
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
        file_name = "cgm_cal"

        mask_start = data["mask_start"]
        mask_end = data["mask_end"]

        agent_input_df = data["window_df"][
            [self.TIMESTAMP_COLUMN, self.CGM_COLUMN, self.CAL_COLUMN]
        ]
        agent_input_data = {f"{file_name}": agent_input_df}

        self.save_data(agent_input_data, query_id=query_id)

        prompt = dedent(f"""\
        The CGM (continuous glucose monitoring) and activity calories consumption data for this subject is provided in 'input/{file_name}.csv'. There are three columns: Timestamp (datetime), 'Libre GL' (CGM, mg/dL), and 'Calories (Activity)' (estimation of calories burned in the past minute by a smartwatch).
        In this {self.window_hours}-hour window, a {self.mask_minutes}-min segment of CGM readings is corrupted (set to 0), start from {mask_start} and end at {mask_end}. Please impute the missing CGM values in this segment using the calories and remaining CGM data. Then save your imputed values in order as a list in json file at `output/solution.json`. When finished, put your methodology to solve this task as your answer.
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
        # Calculate per-sample metrics and average them for accurate evaluation
        # This fixes the concatenation bug where concatenating all predictions treated them as a single long time series
        # Instead, we calculate metrics for each test case individually and then average across all successful cases
        mae_values = []
        mse_values = []
        mape_values = []
        smape_values = []
        minmax_smape_values = []  # for minmax SMAPE
        failure_count = 0
        smape_invalid_count = 0
        minmax_smape_invalid_count = 0  # for minmax SMAPE
        for result in result_list:
            pred = result["solution"]
            gt = result["GT"]
            if pred and isinstance(pred, list) and len(pred) == len(gt):
                # Calculate metrics for this single test case/sample
                sample_mae = Metrics.mae({self.PRED_KEY: pred}, {self.PRED_KEY: gt})[
                    self.PRED_KEY
                ]
                sample_mse = Metrics.mse({self.PRED_KEY: pred}, {self.PRED_KEY: gt})[
                    self.PRED_KEY
                ]
                sample_mape = Metrics.mape({self.PRED_KEY: pred}, {self.PRED_KEY: gt})[
                    self.PRED_KEY
                ]
                smape_result, invalid_count = Metrics.smape(
                    {self.PRED_KEY: pred}, {self.PRED_KEY: gt}
                )
                sample_smape = smape_result[self.PRED_KEY]
                smape_invalid_count += invalid_count

                pred_array = np.array(pred)
                gt_array = np.array(gt)

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
                    gt_array_scaled = (gt_array - common_min) / (
                        common_max - common_min
                    )
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
                sample_minmax_smape = minmax_smape_result[self.PRED_KEY]
                minmax_smape_invalid_count += minmax_invalid_count

                if sample_mae is not None and not np.isnan(sample_mae):
                    mae_values.append(sample_mae)
                if sample_mse is not None and not np.isnan(sample_mse):
                    mse_values.append(sample_mse)
                if sample_mape is not None and not np.isnan(sample_mape):
                    mape_values.append(sample_mape)
                if sample_smape is not None and not np.isnan(sample_smape):
                    smape_values.append(sample_smape)  # NEW
                if sample_minmax_smape is not None and not np.isnan(
                    sample_minmax_smape
                ):
                    minmax_smape_values.append(sample_minmax_smape)  # NEW
            else:
                failure_count += 1
        # Average the per-sample metrics across all successful test cases
        mae = np.mean(mae_values) if mae_values else float("nan")
        mse = np.mean(mse_values) if mse_values else float("nan")
        mape = np.mean(mape_values) if mape_values else float("nan")
        smape = np.mean(smape_values) if smape_values else float("nan")  # NEW
        minmax_smape = (
            np.mean(minmax_smape_values) if minmax_smape_values else float("nan")
        )
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

    def parse_output(self, content=None, query_id=None):
        """
        Parse the agent's output by reading the solution from the JSON file.

        This method reads the imputed CGM values from agent_working/{query_id}/output/solution.json
        using json_repair.load() for robust JSON parsing that can handle minor formatting issues.

        Args:
            content: Optional string content (not used, kept for interface compatibility)
            query_id: Query ID to locate the output file

        Returns:
            List of imputed CGM values, or empty list if parsing fails
        """
        if query_id is None:
            return [], "No query_id provided"

        try:
            solution_path = (
                Path("agent_working") / query_id / "output" / "solution.json"
            )
            with open(solution_path, "r") as f:
                solution = json_repair.load(f)
            return solution
        except Exception as e:
            print(f"Error parsing output for query_id {query_id}: {e}")
            return [], str(e)
