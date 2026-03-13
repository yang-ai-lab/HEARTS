import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from loguru import logger

from agents.base.agent import AgentBase
from exp.utils.datatypes import ExperimentMetrics


class ExperimentBase(ABC):
    """
    Abstract base class for experiments.

    Provides a common structure for running experiments with data preparation,
    iteration, agent execution, output parsing, and metrics calculation.
    """

    def __init__(
        self,
        task: str,
        num_test: int = 50,
        logs_dir: Optional[Path] = None,
        agent: Optional[AgentBase] = None,
    ):
        self.num_test = num_test
        # Set default logs directory if not provided and task is specified
        if logs_dir is not None:
            self.logs_dir = Path(logs_dir) / task
        else:
            self.logs_dir = Path("logs") / task
        if self.logs_dir and not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Logs will be saved to {self.logs_dir}")
        self.task = task
        self.agent = agent

    @abstractmethod
    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent for a single data dict and return the result."""
        pass

    @abstractmethod
    def parse_output(
        self, content: Optional[str] = None, query_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Any]:
        """Parse the output from agent's last response or final answer, or from agent output file if query_id is provided, and return a tuple of (solution dict, fail_reason)."""
        pass

    @abstractmethod
    def calculate_metrics(self, result_list: List[dict]) -> ExperimentMetrics:
        """Calculate and return metrics based on the experiment results."""
        pass

    @abstractmethod
    def save_data(self, data: Dict[str, Any], query_id: Optional[str] = None) -> None:
        """Save input data for agent at specified location that it can access, this should be implemented for each dataset experiment base class."""
        pass

    def cleanup(self, query_id: Optional[str] = None) -> None:
        """Cleanup any temporary files or directories used during the experiment."""
        if query_id is None:
            return
        agent_working_dir = Path("agent_working") / query_id
        if agent_working_dir.exists():
            shutil.rmtree(agent_working_dir)
