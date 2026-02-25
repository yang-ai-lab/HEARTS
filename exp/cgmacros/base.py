import shutil
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from exp.base.base import ExperimentBase
from utils.work_dir import clean_agent_working_dir


class CGMacrosExperiment(ExperimentBase):
    """
    Experiment class for CGMacros dataset, inheriting from ExperimentBase.
    """

    def __init__(
        self,
        task: str,
        num_test: int = 50,
        logs_dir: Optional[Path] = None,
        agent: Optional[Any] = None,
    ):
        super().__init__(task, num_test, logs_dir, agent)

    def save_data(self, agent_input_data: dict, query_id: str):
        """
        Saves input dataframes for an agent to CSV files in a designated working directory.
        This method first cleans the agent's working directory for the given query ID,
        then creates an input directory within it. Each dataframe in `agent_input_data`
        is saved as a CSV file named after its key in the input directory.
        Args:
            agent_input_data (dict[str, pd.DataFrame]):
                A dictionary mapping `name` to DataFrame to be saved.
            query_id (str):
                Identifier for the agent's query, used to determine the working directory.
        """
        clean_agent_working_dir(query_id=query_id)
        agent_working_dir = Path("agent_working") / query_id
        agent_input_dir = agent_working_dir / "input"

        for name, val in agent_input_data.items():
            if isinstance(val, pd.DataFrame):
                df = val
                path = agent_input_dir / f"{name}.csv"
                df.to_csv(path, index=False)
                logger.debug(f"Saved data {name} to {path}")
            elif isinstance(val, str) and val.endswith(".jpg"):
                shutil.copy(val, agent_input_dir / name)
                logger.debug(f"Saved data {val} to {agent_input_dir / name}")
            elif isinstance(val, bytes) and name.endswith(".jpg"):
                # Handle image bytes (for frozen test cases with serialized images)
                path = agent_input_dir / name
                with open(path, "wb") as f:
                    f.write(val)
                logger.debug(f"Saved image bytes to {path}")
        return
