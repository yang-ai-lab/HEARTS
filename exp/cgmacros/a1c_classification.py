import uuid
from textwrap import dedent
from typing import Any, Dict, List
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

@register_experiment("a1c_classification")
class A1CClassificationExperiment(CGMacrosExperiment):
    """
    Experiment class for classifying disease status from CGM data.
    The agent is given the entire CGM time series (timestamps + CGM only) for a subject and must predict the A1C status (e.g., normal, prediabetes, diabetes).
    """
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"
    PRED_KEY = "disease_status"

    def __init__(
        self,
        task: str = "a1c_classification",
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

        # Save agent input data
        self.save_data(agent_input_data, query_id=query_id)

        # Create the prompt
        prompt = dedent(f"""\
        The continuous glucose monitors (CGM) data for this subject is provided in 'input/{file_name}.csv'. There are two columns in this csv file: one is {self.TIMESTAMP_COLUMN} containing the timestamp of each reading, and the other column "{self.CGM_COLUMN}" contains glucose values (mg/dL). Your task is to predict the disease status based on CGM data. There are 3 different status: normal, prediabetes, and diabetes. Please analyze the entire CGM time series and output your final prediction of disease status as a JSON object without any other text in the following format:
        {{
            "{self.PRED_KEY}": [string, normal|prediabetes|diabetes]
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
        Parse the agent's output to extract the predicted disease status.
        Args:
            content: The response content from the agent.
        Returns:
            Dictionary with the parsed status.
        """
        return OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            expected_value_types={self.PRED_KEY: str},
        )

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate accuracy for disease status prediction.
        Args:
            result_list: List of result dictionaries.
        Returns:
            Accuracy and failure count.
        """
        if not result_list:
            return {"Accuracy": float("nan"), "Failures": 0}

        gts = [result["GT"] for result in result_list]
        preds = [
            result["solution"].get(self.PRED_KEY, "NA") if result["solution"] else None
            for result in result_list
        ]
        acc, failure_count = Metrics.acc(preds, gts)
        print("Failures:", failure_count)
        print("Average Accuracy:", acc)
        return {"Accuracy": acc, "Failures": failure_count}
