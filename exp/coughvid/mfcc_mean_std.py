import numpy as np
from exp.utils.datatypes import ExperimentMetrics
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from agents.base.agent import AgentBase
from exp.coughvid.base import CoughVIDExperiment
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

EXPERIMENT_NAME = "mfcc_mean_std"

@register_experiment(EXPERIMENT_NAME)
class MfccMeanStdExperiment(CoughVIDExperiment):
    """
    Experiment class for evaluating MFCC mean and std calculation using an agent.
    """

    PRED_MEAN_KEY = "mfcc_mean"
    PRED_STD_KEY = "mfcc_std"

    def __init__(
        self,
        task: str = EXPERIMENT_NAME,
        num_test: int = 50,
        logs_dir: Optional[Path] = None,
        agent: Optional[AgentBase] = None,
        data_dir: Optional[Path] = None,
    ):
        super().__init__(
            task=task,
            num_test=num_test,
            logs_dir=logs_dir,
            agent=agent,
        )
        self.data_dir = data_dir

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent on the provided data to calculate MFCC mean and std.

        Args:
            data: Dictionary containing subject data including audio and ground truth.

        Returns:
            Dictionary with query results including solution and ground truth.
        """
        # Generate a unique query ID
        query_id = str(uuid.uuid4())

        # Prepare agent input data
        agent_input_data = {"audio": data["audio"], "sr": data["sr"]}
        self.save_data(agent_input_data, query_id=query_id)

        # Create the prompt for the agent
        prompt = dedent(f"""\
        One audio file is saved at 'input/audio.wav'. It is a cough audio sampled at 48 kHz. Calculate the Mel Frequency Cepstral Coefficients (MFCCs) with 13 coefficients, then compute the mean and standard deviation of each MFCC over time. Please output your final answer in the following JSON format without any other text:
        {{
            "{self.PRED_MEAN_KEY}": [list of 13 floats],
            "{self.PRED_STD_KEY}": [list of 13 floats]
        }}""")

        # Query the agent with the prompt and data
        content = await self.agent.query(
            prompt, agent_input_data, self.logs_dir, query_id
        )
        # Parse the agent's output to extract the solution
        solution, fail_reason = self.parse_output(content)
        # Build the result dictionary with all relevant information
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
        Parse the agent's output to extract MFCC mean and std.

        Args:
            content: The response content from the agent.

        Returns:
            Dictionary with mfcc_mean and mfcc_std.
        """

        def validator(solution) -> bool:
            for key in [self.PRED_MEAN_KEY, self.PRED_STD_KEY]:
                value = solution[key]
                assert isinstance(value, list) and len(value) == 13  # 13 MFCCs
                assert all(isinstance(x, (int, float)) for x in value)
            return False

        return OutputStringParser.parse_dict(
            content,
            validator=validator,
        )

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and SMAPE for MFCC mean and std predictions.

        Args:
            result_list: List of result dictionaries.

        Returns:
            Dictionary with average MAE, MAPE, SMAPE for each key and failure count.
        """
        preds_per_key = {self.PRED_MEAN_KEY: [], self.PRED_STD_KEY: []}
        gts_per_key = {self.PRED_MEAN_KEY: [], self.PRED_STD_KEY: []}
        failure_count = 0
        smape_invalid_count = 0
        for result in result_list:
            gt = result["GT"]
            solution = result["solution"]
            if not solution:
                failure_count += 1
                continue
            for key in [self.PRED_MEAN_KEY, self.PRED_STD_KEY]:
                preds_per_key[key].extend(solution[key])
                gts_per_key[key].extend(gt[key])
        mae_dict = Metrics.mae(preds=preds_per_key, gts=gts_per_key)
        mape_dict = Metrics.mape(preds=preds_per_key, gts=gts_per_key)
        # Calculate SMAPE
        smape_dict, invalid_gt_count = Metrics.smape(
            preds=preds_per_key, gts=gts_per_key
        )  # We already accumulated invalid_count above
        smape_invalid_count += invalid_gt_count
        # Calculate overall MAE as average of mfcc_mean and mfcc_std MAE
        overall_mae = float(np.mean(list(mae_dict.values())))
        # Calculate overall MAPE as average of mfcc_mean and mfcc_std MAPE, ignoring infinite values
        mape_values = np.array(list(mape_dict.values()))
        finite_mape_values = mape_values[np.isfinite(mape_values)]
        if finite_mape_values.size > 0:
            overall_mape = float(np.mean(finite_mape_values))
        else:
            overall_mape = float(
                "nan"
            )  # or set to None, or another indicator as appropriate
        # Calculate overall SMAPE as average of mfcc_mean and mfcc_std SMAPE, ignoring infinite values
        smape_values = np.array(list(smape_dict.values()))
        finite_smape_values = smape_values[np.isfinite(smape_values)]
        if finite_smape_values.size > 0:
            overall_smape = float(np.mean(finite_smape_values))
        else:
            overall_smape = float("nan")
        print("Average MAE:", mae_dict)
        print("Average MAPE:", mape_dict)
        print("Average SMAPE:", smape_dict)
        print("Failures:", failure_count)
        return {
            "MAE": overall_mae,
            "MAPE": overall_mape,
            "SMAPE": overall_smape,
            "Failures": failure_count,
            "additional_metrics": {
                "MAE_per_feature": mae_dict,
                "MAPE_per_feature": mape_dict,
                "SMAPE_per_feature": smape_dict,
                "smape_invalid_count": smape_invalid_count,
            },
        }