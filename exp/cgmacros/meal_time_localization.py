import random
import uuid
from collections import defaultdict
from textwrap import dedent
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

@register_experiment("meal_time_localization")
class MealTimeLocalizationExperiment(CGMacrosExperiment):
    """
    Experiment class for localizing meal time from CGM data.
    The agent is given a 2-hour CGM window (timestamps + CGM only) containing exactly one meal event, and must predict the timestamp when the meal starts.
    """

    WINDOW_HOURS = 2
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"
    PRED_KEY = "meal_timestamp"

    def __init__(
        self,
        task: str = "meal_time_localization",
        num_test: int = 50,
        logs_dir=None,
        agent=None,
        data_dir=None,
    ):
        super().__init__(task, num_test, logs_dir, agent, data_dir)

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = str(uuid.uuid4())
        # Save agent input data
        file_name = "cgm"
        # For agent input, use only integer minutes and CGM
        agent_input_df = data["window_df"][["timestamp_min", self.CGM_COLUMN]]
        agent_input_data = {f"{file_name}": agent_input_df}
        self.save_data(agent_input_data, query_id=query_id)
        # Create the prompt
        prompt = dedent(f"""\
        The continuous glucose monitors (CGM) data for this subject is provided in 'input/{file_name}.csv'. There are two columns in this csv file: one is timestamp_min containing the time of each reading (in integer minutes), and the other column "Libre GL" contains glucose values (mg/dL). There is exactly one meal event in this 2-hour window. Please analyze the CGM data and output your final answer as a JSON object without any other text in the following format:
        {{
            "{self.PRED_KEY}": [float, timestamp (in minutes) when the meal starts]
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
            "window_start": data["window_start"],
            "window_end": data["window_end"],
            "GT": int(data["GT"]),
            "solution": solution,
        }

        if fail_reason is not None:
            result["fail_reason"] = fail_reason

        return result

    def parse_output(self, content: str) -> tuple[Dict[str, Any], Any]:
        """
        Parse the agent's output to extract meal timestamp predictions.
        """

        def converter(val):
            if isinstance(val, list) and len(val) > 0:
                return float(val[0])
            return val

        return OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            expected_value_types={self.PRED_KEY: (float, int)},
            value_converter={self.PRED_KEY: converter},
            fallback={},
        )

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate average absolute error, mean absolute percentage error, and SMAPE for meal timestamp prediction using Metrics.mae, Metrics.mape, and Metrics.smape.
        Args:
            result_list: List of result dictionaries.
        Returns:
            MAE, MAPE, SMAPE and failure count.
        """

        preds = defaultdict(list)
        gts = defaultdict(list)
        failure_count = 0
        smape_invalid_count = 0  # NEW
        minmax_smape_invalid_count = 0  # NEW: for minmax SMAPE
        for result in result_list:
            if result["solution"] and self.PRED_KEY in result["solution"]:
                pred_ts = result["solution"][self.PRED_KEY]
                preds[self.PRED_KEY].append(pred_ts)
                gts[self.PRED_KEY].append(result["GT"])
            else:
                failure_count += 1

        if not preds:
            logger.error("No valid predictions were made.")
            return {
                "MAE": float("nan"),
                "MAPE": float("nan"),
                "SMAPE": float("nan"),  # NEW
                "Failures": failure_count,
            }

        mae_dict = Metrics.mae(preds, gts)
        mape_dict = Metrics.mape(preds, gts)
        smape_dict, invalid_count = Metrics.smape(preds, gts)  # NEW
        smape_invalid_count += invalid_count  # NEW

        # NEW: Calculate MINMAX_SMAPE
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

        avg_mae = round(mae_dict[self.PRED_KEY], 4)
        avg_mape = round(mape_dict[self.PRED_KEY], 4)
        avg_smape = round(smape_dict[self.PRED_KEY], 4)  # NEW
        avg_minmax_smape = (  # NEW
            round(float(np.mean(minmax_smape_values)), 4)
            if minmax_smape_values
            else float("nan")
        )
        print("Failures:", failure_count)
        print("Average MAE:", avg_mae)
        print("Average MAPE:", avg_mape)
        print("Average SMAPE:", avg_smape)  # NEW
        print("Average MINMAX_SMAPE:", avg_minmax_smape)  # NEW
        return {
            "MAE": avg_mae,
            "MAPE": avg_mape,
            "SMAPE": avg_smape,  # NEW
            "MINMAX_SMAPE": avg_minmax_smape,  # NEW
            "Failures": failure_count,
            "additional_metrics": {  # NEW
                "smape_invalid_count": smape_invalid_count,
                "minmax_smape_invalid_count": minmax_smape_invalid_count,  # NEW
            },
        }
