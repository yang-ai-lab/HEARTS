import uuid
from textwrap import dedent
from typing import Any, Dict, List
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from exp.vctk.base import VCTKExperiment
from exp.vctk.utils import get_speaker_metadata
from utils.parse_output import OutputStringParser

@register_experiment("waveform_temporal_direction_detection")
class WaveformTemporalDirectionDetectionExperiment(VCTKExperiment):
    """
    Experiment to detect whether a waveform is playing forward or time-reversed.
    """

    def __init__(
        self,
        task: str = "waveform_temporal_direction_detection",
        num_test: int = 50,
        logs_dir=None,
        agent=None,
        target_speakers: List[str] = None,
        sample_rate: int = 16000,
    ):
        super().__init__(
            task=task,
            num_test=num_test,
            logs_dir=logs_dir,
            agent=agent,
            target_speakers=target_speakers,
            sample_rate=sample_rate,
        )

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run agent on a test case."""
        query_id = str(uuid.uuid4())

        # Get speaker metadata
        speaker_metadata = get_speaker_metadata(data["speaker_id"])

        # Prepare input data
        agent_input_data = {
            "waveform": data["waveform"],
            "metadata": {
                "speaker_id": data["speaker_id"],
                "recording_id": data["recording_id"],
                "sample_rate": self.sample_rate,
                "duration_seconds": len(data["waveform"]) / self.sample_rate,
                **speaker_metadata,
            },
        }
        self.save_data(agent_input_data, query_id=query_id)

        prompt = dedent(f"""\
        A raw audio waveform is provided at 'input/audio.npy'. This is a mono audio signal sampled at {self.sample_rate} Hz. The waveform may be playing forward (normal) or time-reversed (backward). Speaker: {data["speaker_id"]} (accent: {speaker_metadata.get("accent", "unknown")}). Determine whether the waveform is playing in the forward direction or has been time-reversed. Output your final answer as a JSON object without any other text, in the following format:
        {{
            "direction": "[forward|reversed]",
            "reason": "[explanation of your choice]"
        }}
        """)

        agent_output = await self.agent.query(
            prompt, data=agent_input_data, logs_dir=self.logs_dir, query_id=query_id
        )

        solution, fail_reason = self.parse_output(agent_output)
        result = {
            "query_id": query_id,
            "speaker_id": data["speaker_id"],
            "recording_id": data["recording_id"],
            "GT": data["GT"],
            "solution": solution,
        }
        if fail_reason is not None:
            result["fail_reason"] = fail_reason
        return result

    def parse_output(self, content: str) -> tuple[Dict[str, Any], Any]:
        """Parse the agent's output."""
        solution, fail_reason = OutputStringParser.parse_dict(
            content,
            expected_keys=["direction"],
            expected_value_types={"direction": str},
            fallback={},
        )

        return solution, fail_reason

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        """Calculate and return metrics based on experiment results."""
        if not result_list:
            return {"Accuracy": float("nan"), "Failures": 0}

        correct = 0
        failures = 0

        for result in result_list:
            gt = result["GT"]
            solution = result["solution"]

            if not solution or solution.get("direction") is None:
                failures += 1
                continue

            direction = solution["direction"].lower()
            pred = 1 if direction == "reversed" else 0

            if pred == gt:
                correct += 1

        total = len(result_list) - failures
        accuracy = correct / total if total > 0 else 0.0
        print(f"acc: {accuracy}, failures: {failures}")
        return {"Accuracy": accuracy, "Failures": failures}
