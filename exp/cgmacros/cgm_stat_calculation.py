"""
CGM statistics calculation (percentage below/above normal range) experiment

supported by: https://diabetes.org/about-diabetes/devices-technology/cgm-time-in-range
"""
import uuid
from collections import defaultdict
from textwrap import dedent
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

@register_experiment("cgm_stat_calculation")
class CGMStatCalculationExperiment(CGMacrosExperiment):
    """
    Experiment class for evaluating CGM statistics calculation using an agent.
    Calculates percentage of time CGM readings are below and above normal range for the entire Libre GL time series.
    """

    NORMAL_RANGE = (70, 180)  # mg/dL, typical CGM normal range

    def __init__(
        self,
        task: str = "cgm_stat_calculation",
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
        agent_input_data = {f"{file_name}": data["cgm"]}
        self.save_data(agent_input_data, query_id=query_id)

        # Create the prompt
        prompt = dedent(f"""\
        The continuous glucose monitors (CGM) data for this subject is provided in 'input/{file_name}.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range ({self.NORMAL_RANGE[0]} - {self.NORMAL_RANGE[1]} mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
        {{
            "below": [float, percentage of time CGM < {self.NORMAL_RANGE[0]} mg/dL],
            "above": [float, percentage of time CGM > {self.NORMAL_RANGE[1]} mg/dL]
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
        Parse the agent's output to extract the statistics.
        Args:
            content: The response content from the agent.
        Returns:
            Dictionary with the parsed statistics.
        """
        return OutputStringParser.parse_dict(
            content,
            expected_keys=["below", "above"],
            expected_value_types={"below": float, "above": float},
        )

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate average Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and SMAPE for percentage predictions.
        Args:
            result_list: List of result dictionaries.
        Returns:
            Average MAE, MAPE, and SMAPE as floats.
        """
        if not result_list:
            return {
                "MAE": float("nan"),
                "MAPE": float("nan"),
                "SMAPE": float("nan"),
                "Failures": 0,
            }  # CHANGED: use NaN instead of 0
        failure_count = 0
        gts = defaultdict(list)
        preds = defaultdict(list)
        smape_invalid_count = 0
        for result in result_list:
            gt = result["GT"]
            solution = result["solution"]
            if not solution:
                failure_count += 1
                continue
            for key in gt.keys():
                preds[key].append(solution[key])  # make sure valid in parse_output
                gts[key].append(gt[key])
        avg_mae = Metrics.mae(preds, gts)
        avg_mape = Metrics.mape(preds, gts)
        smape_result, invalid_count = Metrics.smape(preds, gts)
        smape_invalid_count += invalid_count
        # Calculate overall MAE as average across both metrics
        overall_mae = sum(avg_mae.values()) / len(avg_mae) if avg_mae else float("nan")
        # Calculate overall MAPE as average across both metrics
        finite_values = [v for v in avg_mape.values() if np.isfinite(v)]
        overall_mape = (
            sum(finite_values) / len(finite_values) if finite_values else float("nan")
        )
        # Calculate overall SMAPE as average across both metrics
        finite_smape_values = [v for v in smape_result.values() if np.isfinite(v)]
        overall_smape = (
            sum(finite_smape_values) / len(finite_smape_values)
            if finite_smape_values
            else float("nan")
        )

        print("Failures:", failure_count)
        print("Average MAE:", overall_mae)
        print("Average MAPE:", overall_mape)
        print("Average SMAPE:", overall_smape)
        print("Detailed MAE:", avg_mae)
        print("Detailed MAPE:", avg_mape)
        print("Detailed SMAPE:", smape_result)
        return {
            "MAE": overall_mae,
            "MAPE": overall_mape,
            "SMAPE": overall_smape,
            "Failures": failure_count,
            "additional_metrics": {
                "MAE_per_metric": avg_mae,
                "MAPE_per_metric": avg_mape,
                "SMAPE_per_metric": smape_result,
                "smape_invalid_count": smape_invalid_count,
            },
        }

    def calculate_stat(self, libre_gl: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate CGM statistics from Libre GL time series.
        Args:
            libre_gl: DataFrame containing CGM values (mg/dL) and timestamps
        Returns:
            Dictionary with percentage below/above normal range
        """
        total = len(libre_gl)
        below = sum(1 for v in libre_gl["Libre GL"] if v < self.NORMAL_RANGE[0])
        above = sum(1 for v in libre_gl["Libre GL"] if v > self.NORMAL_RANGE[1])
        below_pct = below / total * 100 if total > 0 else 0
        above_pct = above / total * 100 if total > 0 else 0
        return {
            "below": round(below_pct, 4),
            "above": round(above_pct, 4),
        }
