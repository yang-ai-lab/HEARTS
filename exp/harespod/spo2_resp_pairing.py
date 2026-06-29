import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from agents.base.agent import AgentBase
from exp.harespod.base import HarespodExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

EXPERIMENT_NAME = "spo2_resp_pairing"

@register_experiment(EXPERIMENT_NAME)
class Spo2RespPairingExperiment(HarespodExperiment):
    """
    Experiment to pair corresponding respiration and SpO2 signals from the same subject at different phases.
    """

    PRED_KEY = "pairing"  # Key for pairing dict in agent output
    NUM_PHASES = 2

    def __init__(
        self,
        task: str = EXPERIMENT_NAME,
        num_test: int = 50,
        logs_dir: Optional[Path] = None,
        agent: Optional[AgentBase] = None,
    ):
        super().__init__(
            task=task,
            num_test=num_test,
            logs_dir=logs_dir,
            agent=agent,
        )

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent to pair the respiration and SPO2 signals.

        Args:
            data: Dictionary containing segments and GT.

        Returns:
            Dictionary with query results.
        """
        query_id = str(uuid.uuid4())

        # Prepare agent input data: save the segments
        agent_input_data = {**data["respiration_dfs"], **data["spo_dfs"]}
        self.save_data(agent_input_data, query_id=query_id)

        # Create prompt
        prompt = dedent(f"""\
        You are given 5-minute segments of respiration and SpO2 (oxygen saturation) signals from the same subject at different altitude phases.

        The respiration signals are in input/respiration_A.csv and input/respiration_B.csv, each containing 2 columns: "timestamp" and "respiration".

        The SpO2 signals are in input/spo_1.csv and input/spo_2.csv, each containing 2 columns: "timestamp" and "spo".

        Your task is to analyze the signals and pair which SpO2 signal corresponds to which respiration signal, based on them being from the same altitude phase. The pairing should reflect which SpO2 matches which respiration from the same phase.

        Output your pairing as a dictionary in JSON format without any other text, for example:
        {{
            "{self.PRED_KEY}": {{"A": "1", "B": "2"}} # or {{"A": "2", "B": "1"}}
        }}""")

        content = await self.agent.query(
            prompt, agent_input_data, self.logs_dir, query_id
        )
        solution, fail_reason = self.parse_output(content)

        result = {
            "query_id": query_id,
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

        def validate(solution):
            pairing = solution.get(self.PRED_KEY)
            if not isinstance(pairing, dict) or len(pairing) != self.NUM_PHASES:
                raise ValueError("Invalid pairing format.")
            if set(pairing.keys()) != {"A", "B"}:
                raise ValueError("Invalid respiration labels.")
            if set(pairing.values()) != {"1", "2"}:
                raise ValueError("Invalid SpO2 labels.")
            if len(set(pairing.values())) != self.NUM_PHASES:
                raise ValueError("SpO2 labels not unique.")
            return

        def converter(value):
            if isinstance(value, list) and len(value) == 1:
                return value[0]
            return value

        out = OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            validator=validate,
            value_converter=converter,
        )
        return out

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate accuracy (exact match) and failure count.

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