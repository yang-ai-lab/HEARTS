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

EXPERIMENT_NAME = "altitude_ranking_respiration"

@register_experiment(EXPERIMENT_NAME)
class AltitudeRankingRespirationExperiment(HarespodExperiment):
    """
    Experiment to rank shuffled 5-min respiration segments from different altitude phases
    from high to low altitude.
    """

    PRED_KEY = "ranking"  # Key for ranking list in agent output
    NUM_PHASES = 3

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
        self.subjects_with_valid_phases = []

    async def run_agent(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], Any]:
        """
        Run the agent to rank the shuffled segments by altitude high to low.

        Args:
            data: Dictionary containing segments and GT.

        Returns:
            Dictionary with query results.
        """
        query_id = str(uuid.uuid4())

        # Prepare agent input data: save the segments
        agent_input_data = data["segment_dfs"]
        self.save_data(agent_input_data, query_id=query_id)

        # Create prompt
        prompt = dedent(f"""\
        You are given 5-minute segments of respiration signals at input/[label].csv, which contains 2 columns: "timestamp" and "rsp". Each segment corresponds to one of the altitude ranges provided below:
                        
        - 1.5k-2k meters
        - 2k-2.5k meters
        - 2.5k-3k meters
        - 3k-3.5k meters
        - 3.5k-4k meters
        
        Your task is to analyze the respiration signals and rank the segments from highest altitude to lowest altitude.
        
        Output your ranking as a list of labels in JSON format:
        {{
            "{self.PRED_KEY}": ["label from highest altitude", "label from mid altitude", "label from lowest altitude"]
        }}""")

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

        def validate(solution):
            ranking = solution.get(self.PRED_KEY)
            if not isinstance(ranking, list) or len(ranking) != self.NUM_PHASES:
                raise ValueError("Invalid ranking format.")
            valid_labels = {"A", "B", "C"}
            if set(ranking) != valid_labels:
                raise ValueError("Invalid ranking labels.")
            return

        out = OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            validator=validate,
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