import json
import pickle


def save_state(state, fp):
    with open(fp, "wb") as f:
        pickle.dump(state, f)


def save_json(obj, fp):
    with open(fp, "w") as f:
        json.dump(obj, f, indent=4)


def read_state(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)


def load_json(fp):
    with open(fp, "r") as f:
        return json.load(f)


def extract_reasoning_steps(state):
    """
    Extract reasoning steps from agent state.

    This function extracts the conversation history and code execution steps
    from the agent's state for logging and debugging purposes.

    Args:
        state: Agent state dictionary (typically from LangGraph checkpoint)

    Returns:
        List of reasoning step dictionaries with 'role', 'content', and optionally 'code'
    """
    reasoning_steps = []

    if not state or not isinstance(state, dict):
        return reasoning_steps

    # Extract messages from state
    messages = state.get("messages", [])

    for msg in messages:
        step = {}

        # Get role (user, assistant, system, etc.)
        if hasattr(msg, "role"):
            step["role"] = msg.role
        elif isinstance(msg, dict):
            step["role"] = msg.get("role", "unknown")
        else:
            continue

        # Get content
        if hasattr(msg, "content"):
            step["content"] = str(msg.content)
        elif isinstance(msg, dict):
            step["content"] = str(msg.get("content", ""))
        else:
            step["content"] = str(msg)

        # Extract code if present (from CodeAct agent script execution)
        if "script" in state:
            step["code"] = state.get("script", "")

        reasoning_steps.append(step)

    return reasoning_steps
