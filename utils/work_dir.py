# utilities
import shutil
from pathlib import Path
from typing import Optional


def clean_agent_input_dir(query_id: Optional[str] = None):
    agent_working_dir = Path("agent_working")
    if query_id is not None:
        agent_working_dir = agent_working_dir / query_id
    input_path = agent_working_dir / "input"
    if not input_path.exists():
        input_path.mkdir(exist_ok=True, parents=True)
        return
    shutil.rmtree(input_path)
    input_path.mkdir(exist_ok=True)
    return


def clean_agent_output_dir(query_id: Optional[str] = None):
    agent_working_dir = Path("agent_working")
    if query_id is not None:
        agent_working_dir = agent_working_dir / query_id
    output_path = agent_working_dir / "output"
    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)
        return
    shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    return


def clean_agent_working_dir(query_id: Optional[str] = None):
    clean_agent_input_dir(query_id)
    clean_agent_output_dir(query_id)
    return
