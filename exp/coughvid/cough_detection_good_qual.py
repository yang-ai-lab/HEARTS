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

EXPERIMENT_NAME = "cough_detection_good_qual"

@register_experiment(EXPERIMENT_NAME)
class CoughDetectionExperiment(CoughVIDExperiment):
    """
    Experiment class for evaluating cough detection using an agent.
    The agent is given an audio file and asked to determine if a cough is present.
    Only uses samples where quality is agreed to be 'good' by at least 3 annotators.
    """

    PRED_KEY = "has_cough"

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
        Run the agent to detect if cough is present in the audio.

        Args:
            data: Dictionary containing audio data and ground truth.

        Returns:
            Dictionary with query results.
        """
        query_id = str(uuid.uuid4())

        # Prepare agent input data
        agent_input_data = {"audio": data["audio"], "sr": data["sr"]}
        self.save_data(agent_input_data, query_id=query_id)

        # Create the prompt
        prompt = dedent(f"""\
        One audio file is saved at 'input/audio.wav'. It is a cough audio sampled at 48 kHz. Analyze the audio and determine if a cough is present in the recording.

        Output your answer in the following JSON format without any other text:
        {{
            "{self.PRED_KEY}": true/false
        }}""")

        # Query the agent
        content = await self.agent.query(
            prompt, agent_input_data, self.logs_dir, query_id
        )
        solution, fail_reason = self.parse_output(content)

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
        Parse the agent's output.

        Args:
            content: Agent response.

        Returns:
            Parsed solution.
        """
        return OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            expected_value_types={self.PRED_KEY: bool},
        )

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate accuracy and failure count.

        Args:
            result_list: List of results.

        Returns:
            Dictionary with accuracy and failure count.
        """
        preds = [result["solution"].get(self.PRED_KEY) for result in result_list]
        gts = [result["GT"] for result in result_list]
        accuracy, failure_count = Metrics.acc(preds, gts)
        print(f"Accuracy: {accuracy}, Failures: {failure_count}")
        return {"Accuracy": accuracy, "Failures": failure_count}