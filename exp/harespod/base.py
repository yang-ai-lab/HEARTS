from pathlib import Path
from typing import Optional

import pandas as pd

from exp.base.base import ExperimentBase
from utils.work_dir import clean_agent_working_dir


class HarespodExperiment(ExperimentBase):
    def __init__(
        self,
        task,
        num_test=50,
        logs_dir=None,
        agent=None,
    ):
        super().__init__(task, num_test, logs_dir, agent)

    def save_data(self, agent_input_data: dict, query_id: Optional[str] = None):
        """Save the data in the required format."""
        clean_agent_working_dir(query_id=query_id)
        agent_working_dir = Path("agent_working")
        if query_id is not None:
            agent_working_dir = agent_working_dir / query_id
        agent_input_dir = agent_working_dir / "input"
        agent_input_dir.mkdir(parents=True, exist_ok=True)

        for key in agent_input_data:
            data = agent_input_data[key]
            if isinstance(data, pd.DataFrame):
                data.to_csv(agent_input_dir / f"{key}.csv", index=False)
        return
