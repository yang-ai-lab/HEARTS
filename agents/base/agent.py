from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class AgentBase(ABC):
    """
    Abstract base class for all agents in the benchmark.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.name = self.__class__.__name__

    @abstractmethod
    async def query(
        self, prompt: str, data: Dict[str, Any], logs_dir: Path, query_id: str
    ) -> str:
        """
        Run the agent on the given prompt and data.

        Args:
            prompt: The prompt to send to the agent.
            data: Dictionary containing data for the query (e.g., signals, channels).
            logs_dir: Directory to save logs and conversation history.
            query_id: Unique identifier for this query.

        Returns:
            The last response content from the agent.
        """
        pass
