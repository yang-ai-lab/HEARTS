## Overview

Agents in this project are implementations that can process prompts and data to generate responses. They follow a standardized interface defined by the `AgentBase` abstract class.

## Structure

Agents are organized in the `agents/` directory. Each agent has its own subdirectory containing:

- `agent.py`: The main agent class implementation
- Additional implementation files as needed

## Base Class

All agents must inherit from `AgentBase` located in `agents/base/agent.py`. The base class defines the following interface:

- `query(prompt: str, data: Dict[str, Any], logs_dir: Path, query_id: str) -> str`: Asynchronous method to process a prompt and data, returning the agent's last response as final answer.

## Steps to Add a New Agent

### 1. Create Agent Directory

Create a new directory under `agents/` for your agent:

```bash
mkdir agents/your_agent_name
```

### 2. Implement the Agent Class

Create `agents/your_agent_name/agent.py` with your agent implementation:

```python
from typing import Dict, Any
from pathlib import Path
from agents.base.agent import AgentBase
from exp.utils.registry import register_agent

@register_agent("your_agent_name")
class YourAgentNameAgent(AgentBase):
    """
    Your agent implementation.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        # Initialize your agent-specific parameters
        self.some_param = kwargs.get('some_param', 'default_value')

    async def query(
        self, prompt: str, data: Dict[str, Any], logs_dir: Path, query_id: str
    ) -> str:
        """
        Run your agent on the given prompt and data.

        Args:
            prompt: The prompt to send to the agent.
            data: Dictionary containing data (e.g., signals, channels).
            logs_dir: Directory to save logs.
            query_id: Unique identifier for this query.

        Returns:
            The last response content.
        """
        # Your agent logic here
        # Process the prompt and data
        # Return the response as a string
        return "Your agent's response"
```

### 3. Register the Agent

The `@register_agent("your_agent_name")` decorator automatically registers your agent in the global registry. The name you provide here will be used in the configuration.

### 4. Handle Data (if applicable)

If your agent needs to process specific data formats (like the CodeAct agent handles signals), implement the data preprocessing in the `query` method:

```python
# Example from CodeAct agent
if "signals" in data and "channels" in data and "mask" in data:
    # Process signals data
    for ch in data["channels"]:
        # Save or process channel data
```

### 5. Logging and State Saving

Use the `logs_dir` and `query_id` to save logs and state for debugging and analysis.

```python
import pickle
from utils.save_log import save_state

# Save agent state (conversation history, executed code, etc.)
state = {"messages": [...], "other_data": ...}
save_state(state, logs_dir / f"{query_id}.pkl")
```

### 6. Configuration

Update `config.yaml` to use your new agent:

```yaml
agent:
  name: "your_agent_name"
  params:
    some_param: "value"
```

## Example: CodeAct Agent

The existing CodeAct agent (`agents/codeact/`) demonstrates a complete implementation:

- **agent.py**: Main agent class with data preprocessing and state saving
- **codeact_impl.py**: Core CodeAct logic using LangGraph and Jupyter kernel
- **utils/agent.py**: Jupyter kernel implementation and agent utilities

Key features:
- Uses LangChain and LangGraph for agent orchestration
- Executes Python code in a Jupyter kernel
- Handles structured output with `<thought>`, `<execute>`, and `<solution>` tags
- Saves conversation state for debugging

## Testing

After implementing your agent:

1. Update `config.yaml` with your agent configuration
2. Run a test experiment using `uv`:
   ```bash
   uv run python run_exp_freeze.py
   ```
3. Check logs in the `logs/` directory (or configured `logs_dir`) for debugging
4. Verify results in the `results/` directory (or configured `result_dir`)

## Contributing

When contributing a new agent:

1. Follow the naming convention: `YourAgentNameAgent`
2. Add comprehensive docstrings
3. Include example usage in comments
4. Test thoroughly before submitting
5. Format and lint your code using `ruff` before committing:
   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   ```
