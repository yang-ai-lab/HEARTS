import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import typer
import yaml
from loguru import logger

from utils.save_log import read_state

app = typer.Typer()


class KernelInterruptExperiment(TypedDict):
    """Type definition for kernel interrupt experiment information."""

    experiment_number: int
    has_kernel_interrupt: bool
    config_start_line: int
    dataset_name: str
    task: str
    kernel_interrupt_found: bool
    model_name: str
    result_dir: str
    logs_dir: str


def parse_config_dict(config_lines: List[str]) -> Dict[str, Any]:
    """
    Parse the config dict from the log lines.

    Args:
        config_lines: Lines containing the config dict

    Returns:
        Parsed config dictionary, or empty dict if parsing fails
    """
    config_text = "".join(config_lines)
    try:
        # Try to find the dict in the text - it should be a Python dict literal
        # Look for the opening brace and find the matching closing brace
        start_idx = config_text.find("{")
        if start_idx == -1:
            return {}

        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(config_text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if brace_count != 0:
            return {}

        dict_str = config_text[start_idx : end_idx + 1]
        # Use ast.literal_eval for safe evaluation of Python literals
        return ast.literal_eval(dict_str)
    except Exception:
        return {}


def extract_kernel_interrupt_experiments(
    log_file_path: str,
    consecutive_thres: int = 3,
) -> List[KernelInterruptExperiment]:
    """
    Extract experiments that contain 2 or more consecutive kernel interrupt logs.

    Args:
        log_file_path: Path to the log file to analyze
        consecutive_thres: Minimum number of consecutive kernel interrupts to flag an experiment

    Returns:
        List of dicts containing experiment info and whether it has kernel interrupts
    """
    experiments = []

    try:
        with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            current_experiment = None
            experiment_count = 0
            consecutive_interrupts = 0
            has_consecutive_interrupts = False
            config_lines = []
            collecting_config = False

            for line_num, line in enumerate(f, 1):
                # Check if this is the start of a new experiment
                # Pattern: timestamp | INFO | utils.exp:load_config:<number> -
                if re.search(r"INFO\s+.*utils\.exp:load_config:\d+\s*-", line):
                    # Save previous experiment if it exists
                    if current_experiment is not None:
                        config = parse_config_dict(config_lines)
                        experiments.append(
                            {
                                "experiment_number": experiment_count,
                                "has_kernel_interrupt": has_consecutive_interrupts,
                                "config_start_line": current_experiment,
                                "dataset_name": config.get("dataset_name", "unknown"),
                                "task": config.get("task", "unknown"),
                                "kernel_interrupt_found": has_consecutive_interrupts,
                                "model_name": config.get("model_name", "unknown_model"),
                                "result_dir": config.get(
                                    "result_dir", "unknown_result_dir"
                                ),
                                "logs_dir": config.get("logs_dir", "unknown_logs_dir"),
                            }
                        )

                    # Start new experiment
                    experiment_count += 1
                    current_experiment = line_num
                    consecutive_interrupts = 0
                    has_consecutive_interrupts = False
                    config_lines = []
                    collecting_config = True

                elif current_experiment is not None:
                    # Collect config lines (typically the next few lines after the INFO log)
                    if collecting_config and line.strip():
                        config_lines.append(line.strip())
                        # Stop collecting after we get a reasonable amount of config
                        if len(config_lines) > 20:  # Config dict should be much shorter
                            collecting_config = False

                    # Check for kernel interrupt within current experiment
                    if "Kernel interrupt" in line or "Kernel interrupted" in line:
                        consecutive_interrupts += 1
                        if consecutive_interrupts >= consecutive_thres:
                            has_consecutive_interrupts = True
                            collecting_config = False  # Stop collecting config once we find consecutive interrupts
                    else:
                        # Reset consecutive counter if we find a non-interrupt line
                        consecutive_interrupts = 0

            # Don't forget the last experiment
            if current_experiment is not None:
                config = parse_config_dict(config_lines)
                experiments.append(
                    {
                        "experiment_number": experiment_count,
                        "has_kernel_interrupt": has_consecutive_interrupts,
                        "config_start_line": current_experiment,
                        "dataset_name": config.get("dataset_name", "unknown"),
                        "task": config.get("task", "unknown"),
                        "kernel_interrupt_found": has_consecutive_interrupts,
                        "model_name": config.get("model_name", "unknown_model"),
                        "result_dir": config.get("result_dir", "unknown_result_dir"),
                        "logs_dir": config.get("logs_dir", "unknown_logs_dir"),
                    }
                )

    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)

    return experiments


def yaml_print_out(experiments: List[KernelInterruptExperiment]):
    tmp_dict = defaultdict(list)
    for d in experiments:
        if d["has_kernel_interrupt"]:
            tmp_dict[d["dataset_name"]].append(d["task"])
    print(f"Model: {d['model_name']}")
    print(yaml.dump(dict(tmp_dict)))
    return


def get_states_with_interrupts(
    experiments: List[KernelInterruptExperiment],
    timeout_thres: int = 2,
) -> List[str]:
    """
    Get the list of query IDs that had kernel interrupts.

    Args:
        experiments: List of experiment dicts

    Returns:
        List of query IDs with kernel interrupts
    """
    # dataset & task combinations interested in
    dataset_task_dict = defaultdict(list)
    for exp in experiments:
        if exp["has_kernel_interrupt"]:
            dataset_name = exp["dataset_name"]
            task = exp["task"]
            dataset_task_dict[dataset_name].append(task)
    result_dir = experiments[0]["result_dir"]
    logs_dir = experiments[0]["logs_dir"]

    logger.info(f"result_dir: {result_dir}, logs_dir: {logs_dir}")

    # find result files
    out = defaultdict(list)
    result_dir = Path(result_dir)
    for dataset in dataset_task_dict:
        for task in dataset_task_dict[dataset]:
            task_log_dir = Path(logs_dir) / task
            logger.info(f"Processing dataset: {dataset}, task: {task}")
            file_pattern = f"{dataset}_{task}_*.json"
            file_paths = list(result_dir.glob(file_pattern))
            logger.info(
                f"Found {len(file_paths)} result files for {dataset}/{task}, choose latest one"
            )
            file_paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            result_fp = file_paths[0]
            logger.info(f"Loading result file: {result_fp}")
            with open(result_fp, "r") as f:
                result_data = json.load(f)
            for item in result_data[:-1]:
                qid = item.get("query_id")
                if not qid:
                    logger.warning(f"No query_id found in item: {item}")
                    continue
                log_fp = task_log_dir / f"{qid}.pkl"
                state = read_state(log_fp)
                msgs = state.get("messages", [])

                timeout_count = 0

                for msg in msgs:
                    if msg.type == "human":
                        if isinstance(msg.content, str) and "timed out" in msg.content:
                            timeout_count += 1
                        if isinstance(msg.content, list):
                            for c in msg.content:
                                if c.get("type") == "text" and "timed out" in c.get(
                                    "text", ""
                                ):
                                    timeout_count += 1
                    if timeout_count >= timeout_thres:
                        out[f"{dataset}/{task}"].append(state)
                        break
    return out


@app.command("ei")
def extract_kernel_interrupts(
    log_file_path: str = typer.Argument(..., help="Path to the log file to analyze"),
    consecutive_thres: int = typer.Option(
        3, help="Minimum number of consecutive kernel interrupts to flag an experiment"
    ),
):
    """
    Extract and display experiments that contain kernel interrupts from a log file.
    """
    experiments = extract_kernel_interrupt_experiments(log_file_path, consecutive_thres)

    # Filter to only experiments with kernel interrupts
    interrupt_experiments = [exp for exp in experiments if exp["has_kernel_interrupt"]]

    if not interrupt_experiments:
        typer.echo("No experiments with kernel interrupts found.")
        return

    typer.echo(
        f"Found {len(interrupt_experiments)} experiment(s) with kernel interrupts:"
    )
    typer.echo()

    for exp in interrupt_experiments:
        dataset_name = exp["dataset_name"]
        task = exp["task"]
        typer.echo(f"{dataset_name}: {task}")


@app.command("dummy")
def dummy_command():
    typer.echo("This is a dummy command.")


if __name__ == "__main__":
    app()
    # uv run utils/debug.py ei <log_file_path>
