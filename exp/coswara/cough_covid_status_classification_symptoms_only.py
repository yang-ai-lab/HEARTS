import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional
from agents.base.agent import AgentBase
from exp.coswara.base import CoswaraExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

EXPERIMENT_NAME = "cough_covid_status_classification_symptoms_only"

@register_experiment(EXPERIMENT_NAME)
class CoughCovidStatusClassificationSymptomsOnlyExperiment(CoswaraExperiment):
    """
    Experiment to classify subjects as healthy or covid positive based on symptoms only.
    """

    PRED_KEY = "prediction"

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
        self.cough_types = ["cough-heavy", "cough-shallow"]

    def _extract_symptoms(self, annotation: Dict[str, Any]) -> Dict[str, bool]:
        """
        Extract symptoms from annotation.
        """
        symptom_cols = [
            "cough",
            "fever",
            "cold",
            "diarrhoea",
            "loss_of_smell",
            "mp",
            "st",
            "bd",
            "ftg",
        ]
        symptoms = {}
        for symptom in symptom_cols:
            value = annotation.get(symptom)
            if value is not None:
                symptoms[symptom] = bool(value)
            else:
                symptoms[symptom] = False  # or None, but for simplicity
        return symptoms

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent to classify the subject's covid status based on symptoms.
        """
        query_id = str(uuid.uuid4())
        # No audio to save
        agent_input_data = {}

        # Format symptoms for prompt
        symptoms_text = self._format_symptoms(data["symptoms"])

        prompt = dedent(f"""\
        Analyze the subject's symptoms to classify the subject as either healthy or covid positive.

        Symptoms: {symptoms_text}

        Based on the symptoms, classify the subject as either healthy or covid positive.

        Output your classification in JSON format:
        {{
            "{self.PRED_KEY}": "healthy" or "covid_positive"
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

    def _format_symptoms(self, symptoms: Dict[str, bool]) -> str:
        """
        Format symptoms dictionary into readable text.
        """
        if not symptoms:
            return "No symptom information available."
        symptom_names = {
            "cough": "Cough",
            "fever": "Fever",
            "cold": "Cold",
            "diarrhoea": "Diarrhoea",
            "loss_of_smell": "Loss of smell",
            "mp": "Muscle pain",
            "st": "Sore throat",
            "bd": "Breathing difficulty",
            "ftg": "Fatigue",
        }
        parts = []
        for key, value in symptoms.items():
            name = symptom_names.get(key, key)
            status = "yes" if value else "no"
            parts.append(f"{name}: {status}")
        return ", ".join(parts)

    def parse_output(self, content: str) -> tuple[Dict[str, Any], Any]:
        """
        Parse the agent's output.
        """

        def validate(solution: dict):
            assert solution[self.PRED_KEY] in ["healthy", "covid_positive"], (
                f"Invalid prediction: {solution[self.PRED_KEY]}"
            )

        out = OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            expected_value_types={self.PRED_KEY: str},
            validator=validate,
        )
        return out

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """
        Calculate accuracy and failure count.
        """
        preds = [result["solution"].get(self.PRED_KEY) for result in result_list]
        gts = [result["GT"] for result in result_list]
        accuracy, failure_count = Metrics.acc(preds, gts)
        print(f"Accuracy: {accuracy}, Failures: {failure_count}")
        return {"Accuracy": accuracy, "Failures": failure_count}