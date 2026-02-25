import uuid
from textwrap import dedent
from typing import Any, Dict, List
import numpy as np
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

@register_experiment("fasting_glu_prediction")
class FastingGluPredictionExperiment(CGMacrosExperiment):
    """
    Experiment class for predicting fasting blood glucose from CGM data.
    The agent is given the entire CGM time series (timestamps + CGM only) for a subject and must predict the fasting blood glucose value (mg/dL).
    """
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"
    PRED_KEY = "fasting_glu"

    def __init__(
        self,
        task: str = "fasting_glu_prediction",
        num_test: int = 50,
        logs_dir=None,
        agent=None,
        data_dir=None,
    ):
        super().__init__(task, num_test, logs_dir, agent, data_dir)

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = str(uuid.uuid4())
        file_name = "cgm"
        # For agent input, use only timestamps and CGM
        agent_input_df = data["window_df"][[self.TIMESTAMP_COLUMN, self.CGM_COLUMN]]
        agent_input_data = {f"{file_name}": agent_input_df}
        self.save_data(agent_input_data, query_id=query_id)
        # Create the prompt
        prompt = dedent(f"""\
        The continuous glucose monitors (CGM) data for this subject is provided in 'input/{file_name}.csv'. There are two columns in this csv file: one is {self.TIMESTAMP_COLUMN} containing the timestamp of each reading, and the other column "{self.CGM_COLUMN}" contains glucose values (mg/dL). Your task is to predict the subject's fasting blood glucose value (in mg/dL) based on the entire CGM time series. Please output your final prediction as a JSON object without any other text in the following format:
        {{
            "{self.PRED_KEY}": [float, fasting blood glucose value in mg/dL]
        }}""")
        # Query the agent
        agent_output = await self.agent.query(
            prompt,
            agent_input_data,
            self.logs_dir,
            query_id=query_id,
        )
        solution, fail_reason = self.parse_output(agent_output)
        result = {
            "query_id": query_id,
            "subject_id": data["subject_id"],
            "GT": data["GT"],
            "solution": solution,
        }

        if fail_reason is not None:
            result["fail_reason"] = fail_reason

        return result

    def parse_output(self, content: str) -> tuple[Dict[str, Any], Any]:
        """
        Parse the agent's output to extract the predicted fasting glucose value.
        Args:
            content: The response content from the agent.
        Returns:
            Dictionary with the parsed value.
        """

        def converter(value):
            if isinstance(value, list) and len(value) == 1:
                return value[0]
            return value

        return OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            value_converter=converter,
            expected_value_types={self.PRED_KEY: float},
        )

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate MAE, MAPE, and SMAPE for fasting glucose prediction.
        Args:
            result_list: List of result dictionaries.
        Returns:
            MAE, MAPE, SMAPE and failure count.
        """
        if not result_list:
            return {
                "MAE": float("nan"),
                "MAPE": float("nan"),
                "SMAPE": float("nan"),
                "Failures": 0,
            }  # CHANGED: use NaN instead of 0
        preds = {self.PRED_KEY: []}
        gts = {self.PRED_KEY: []}
        failure_count = 0
        smape_invalid_count = 0
        minmax_smape_invalid_count = 0 
        for result in result_list:
            pred_val = (
                result["solution"].get(self.PRED_KEY) if result["solution"] else None
            )
            if pred_val is not None:
                try:
                    preds[self.PRED_KEY].append(float(pred_val))
                    gts[self.PRED_KEY].append(float(result["GT"]))
                except Exception:
                    failure_count += 1
            else:
                failure_count += 1
        mae_dict = Metrics.mae(preds, gts)
        mape_dict = Metrics.mape(preds, gts)
        smape_dict, invalid_count = Metrics.smape(preds, gts)
        smape_invalid_count += invalid_count

        minmax_smape_values = []
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
            minmax_smape_invalid_count += minmax_invalid_count
            minmax_smape_values.append(minmax_smape_result[self.PRED_KEY])

        avg_mae = (
            round(mae_dict[self.PRED_KEY], 4) if preds[self.PRED_KEY] else float("nan")
        )
        avg_mape = (
            round(mape_dict[self.PRED_KEY], 4) if preds[self.PRED_KEY] else float("nan")
        )
        avg_smape = (
            round(smape_dict[self.PRED_KEY], 4)
            if preds[self.PRED_KEY]
            else float("nan")
        )
        avg_minmax_smape = (
            round(float(np.mean(minmax_smape_values)), 4)
            if minmax_smape_values
            else float("nan")
        )
        print("Failures:", failure_count)
        print("Average MAE:", avg_mae)
        print("Average MAPE:", avg_mape)
        print("Average SMAPE:", avg_smape)
        print("Average MINMAX_SMAPE:", avg_minmax_smape)
        return {
            "MAE": avg_mae,
            "MAPE": avg_mape,
            "SMAPE": avg_smape,
            "MINMAX_SMAPE": avg_minmax_smape,
            "Failures": failure_count,
            "additional_metrics": {
                "smape_invalid_count": smape_invalid_count,
                "minmax_smape_invalid_count": minmax_smape_invalid_count,
            },
        }
