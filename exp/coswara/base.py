from loguru import logger
from scipy.io.wavfile import write

from exp.base import ExperimentBase
from utils.work_dir import clean_agent_working_dir


class CoswaraExperiment(ExperimentBase):
    def save_data(self, agent_input_data, query_id: str) -> None:
        """Save the data in the required format."""
        # clean it first :)
        clean_agent_working_dir(query_id=query_id)
        if "audio" in agent_input_data:
            audio = agent_input_data["audio"]
            sr = agent_input_data.get("sr", 48000)
            path = f"agent_working/{query_id}/input/audio.wav"
            write(path, sr, audio)
            logger.debug(f"Saved audio to {path}")
