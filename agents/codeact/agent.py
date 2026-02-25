from pathlib import Path
from typing import Any, Dict

from loguru import logger

from agents.base.agent import AgentBase
from agents.codeact.codeact_impl import query_codeact
from exp.utils.registry import register_agent
from utils.model_enums import TEXT_ONLY_MODELS
from utils.save_log import save_state


@register_agent("codeact")
class CodeActAgent(AgentBase):
    """
    CodeAct agent implementation.
    """

    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        only_thoughts_limit: int = 30,
        use_multimodal: bool = False,
        timeout: int = 120,
    ):
        super().__init__(model_name)
        self.verbose = verbose
        self.only_thoughts_limit = only_thoughts_limit
        self.use_multimodal = use_multimodal
        self.timeout = timeout
        # Disable use_multimodal for text-only models
        if self.use_multimodal:
            for text_only_model in TEXT_ONLY_MODELS:
                if text_only_model in model_name:
                    logger.warning(
                        f"Model {model_name} does not support multimodal input. Disabling use_multimodal."
                    )
                    self.use_multimodal = False

    async def query(
        self,
        prompt: str,
        data: Dict[str, Any],
        logs_dir: Path,
        query_id: str,
    ) -> str:
        """
        Run the CodeAct agent on the given prompt and data.

        Args:
            prompt: The prompt to send to the agent.
            data: Dictionary containing data (e.g., signals, channels, mask).
            logs_dir: Directory to save logs.
            query_id: Unique identifier for this query.

        Returns:
            The last response content.
        """
        state = await query_codeact(
            prompt,
            self.model_name,
            verbose=self.verbose,
            query_id=query_id,
            only_thoughts_limit=self.only_thoughts_limit,
            use_multimodal=self.use_multimodal,
            timeout=self.timeout,
        )

        # Save the agent's state to a pickle file in the logs directory
        save_state(state, logs_dir / f"{query_id}.pkl")

        # Return final answer if available, else return the last message content
        return state.get("final_answer", state["messages"][-1].content)
