"""
Base experiment class for VCTK tasks.

Handles raw waveform audio data instead of pre-computed spectrograms.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from exp.base import ExperimentBase
from utils.work_dir import clean_agent_working_dir


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class VCTKExperiment(ExperimentBase):
    """
    Base class for VCTK experiments.

    Extends ExperimentBase with VCTK-specific data handling for raw waveforms.
    """

    def __init__(
        self,
        task: str,
        num_test: int = 50,
        logs_dir: Optional[Path] = None,
        agent=None,
        target_speakers: Optional[list] = None,
        sample_rate: int = 16000,
    ):
        """
        Initialize VCTK experiment.

        Args:
            task: Task name
            num_test: Number of test cases
            logs_dir: Directory for logs
            agent: Agent instance
            data_dir: Path to VCTK dataset directory
            target_speakers: Optional list of speaker IDs to filter to
            sample_rate: Target sample rate for audio (default 16000 Hz)
        """
        super().__init__(task, num_test, logs_dir, agent)
        self.target_speakers = target_speakers
        self.sample_rate = sample_rate

    def filter_by_speakers(self, recordings: list) -> list:
        """
        Filter recordings to only include target speakers.

        Args:
            recordings: List of recording dictionaries

        Returns:
            Filtered list (all recordings if no target_speakers set)
        """
        if self.target_speakers is None:
            return recordings

        return [r for r in recordings if r.get("speaker_id") in self.target_speakers]

    def get_waveform_description(self, sample_rate: int = 16000) -> str:
        """
        Get standardized waveform description for prompts.

        Args:
            sample_rate: Sample rate in Hz (default 16000)

        Returns:
            Formatted description string
        """
        return (
            f"A raw audio waveform is provided at 'input/audio.npy'. "
            f"This is a mono audio signal sampled at {sample_rate} Hz. "
            f"The waveform is stored as a 1D numpy array of floating-point values."
        )

    def save_data(
        self, agent_input_data: Dict[str, Any], query_id: Optional[str] = None
    ):
        """
        Save VCTK data in format expected by agent.

        Saves raw waveform, transcript, and metadata to agent_working/{query_id}/input/ directory.

        Args:
            agent_input_data: Dictionary with waveform data
            query_id: Optional unique identifier for this query (used for directory isolation)
        """
        # Clean working directory first
        clean_agent_working_dir(query_id=query_id)

        # Set up base directory
        agent_working_dir = Path("agent_working")
        if query_id is not None:
            agent_working_dir = agent_working_dir / query_id
        agent_input_dir = agent_working_dir / "input"
        agent_input_dir.mkdir(parents=True, exist_ok=True)

        # Save raw waveform
        if "waveform" in agent_input_data:
            path = agent_input_dir / "audio.npy"
            waveform = agent_input_data["waveform"]
            np.save(path, waveform)
            duration = len(waveform) / self.sample_rate
            logger.debug(
                f"Saved waveform with {len(waveform)} samples ({duration:.2f}s) to {path}"
            )

        # Save F0 trajectory if provided (for gap filling task)
        if "f0_trajectory" in agent_input_data:
            # Already saved as numpy file in the task, but ensure it's there
            if isinstance(agent_input_data["f0_trajectory"], np.ndarray):
                path = agent_input_dir / "f0_trajectory.npy"
                np.save(path, agent_input_data["f0_trajectory"])
                logger.debug(
                    f"Saved F0 trajectory with {len(agent_input_data['f0_trajectory'])} frames to {path}"
                )

        # Save transcript if available (for reference, but agent shouldn't see ground truth)
        # Note: In actual experiments, we typically don't save the ground truth transcript
        # to avoid the agent seeing it. This is here for debugging/development.
        if "transcript" in agent_input_data and agent_input_data.get(
            "include_transcript_in_input", False
        ):
            path = agent_input_dir / "transcript.txt"
            with open(path, "w") as f:
                f.write(agent_input_data["transcript"])
            logger.debug(f"Saved transcript to {path} (for reference only)")

        # Save metadata as JSON
        if "metadata" in agent_input_data:
            metadata = agent_input_data["metadata"].copy()

            # Add audio parameters if not already present
            if "audio_params" not in metadata:
                metadata["audio_params"] = {
                    "sample_rate": self.sample_rate,
                    "duration_seconds": len(agent_input_data.get("waveform", []))
                    / self.sample_rate,
                    "num_samples": len(agent_input_data.get("waveform", [])),
                    "format": "mono",
                    "dtype": "float32",
                }

            # Convert numpy types to native Python types for JSON serialization
            metadata = convert_numpy_types(metadata)

            path = agent_input_dir / "metadata.json"
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata to {path}")
