"""
utils for run_exp.py
"""

import asyncio
import datetime
import importlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from agents.base.agent import AgentBase
from exp.base.base import ExperimentBase
from exp.utils.registry import get_agent, get_experiment
from utils.model_enums import AWSBedrockModelNames, ModelNames, OpenAIModelNames, GeminiModelNames, XAIModelNames


@dataclass
class AgentConfig:
    name: str = "codeact"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    dataset_name: str
    task: TaskConfig
    num_test: int
    model_name: str
    agent: AgentConfig
    fix_test_cases_dir: Optional[str] = field(default=None)
    result_dir: Optional[str] = field(default="results")
    logs_dir: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)


class LogLevel(Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def _complete_model_name(model_name: str) -> str:
    if model_name in [e.value for e in OpenAIModelNames]:
        return "openai:" + model_name
    elif model_name in [e.value for e in AWSBedrockModelNames]:
        return "bedrock_converse:" + model_name
    elif model_name in [e.value for e in GeminiModelNames]:
        return "google_genai:" + model_name # use this prefix if you want to use gemini api
        # return "google_vertexai:" + model_name # use this prefix to use vertex ai api
    elif model_name in [e.value for e in XAIModelNames]:
        return "xai:" + model_name
    else:
        raise ValueError(f"Model name '{model_name}' is not valid.")


def load_config(
    config_path: str = "config.yaml", overrides: Optional[dict[str, Any]] = None
) -> Config:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config_dict: dict = yaml.safe_load(f)
    if overrides is not None:
        config_dict.update(overrides)

    logger.info("\n" + pformat(config_dict))

    if config_dict["agent"]["name"] != "biomni": # biomni agent doesn't need prefix
        config_dict["model_name"] = _complete_model_name(config_dict["model_name"])

    # Extract task config from nested 'task' field
    task_field = config_dict.get("task", {})
    if isinstance(task_field, dict):
        task_config = TaskConfig(
            name=task_field.get("name", ""), params=task_field.get("params", {})
        )
    elif isinstance(task_field, str):
        task_config = TaskConfig(name=task_field)

    # make sure fix_test_cases_dir exists if provided
    fix_test_cases_dir = config_dict.get("fix_test_cases_dir")
    if fix_test_cases_dir is not None:
        fix_test_cases_dir = str(fix_test_cases_dir)
        if not Path(fix_test_cases_dir).exists():
            raise ValueError(
                f"fix_test_cases_dir '{fix_test_cases_dir}' does not exist"
            )

    # NOTE: by default "./results/"
    result_dir = config_dict.get("result_dir") or "results"
    logs_dir = config_dict.get("logs_dir") or "logs"

    # Resolve paths to absolute paths
    result_dir = str(Path(result_dir).resolve())
    logs_dir = str(Path(logs_dir).resolve())

    return Config(
        dataset_name=config_dict["dataset_name"],
        task=task_config,
        num_test=config_dict["num_test"],
        model_name=config_dict["model_name"],
        agent=AgentConfig(**config_dict.get("agent", {})),
        fix_test_cases_dir=fix_test_cases_dir,
        result_dir=result_dir,
        logs_dir=logs_dir,
    )


def get_agent_instance(agent_config: AgentConfig, model_name: str) -> AgentBase:
    agent_name = agent_config.name
    # Import the module to register the agent
    importlib.import_module(f"agents.{agent_name}.agent")
    agent_class = get_agent(agent_name)
    if agent_class is None:
        raise ValueError(f"Agent '{agent_name}' not found in registry")
    params = agent_config.params
    agent = agent_class(model_name=model_name, **params)
    print(f"✅ Agent class loaded: {agent_class.__name__}")
    return agent


def get_experiment_class(
    dataset_name: str,
    task: str,
    num_test: int,
    agent: AgentBase,
    params: dict = None,
    logs_dir: Optional[str] = None,
) -> ExperimentBase:
    # Import the module to register the experiment
    importlib.import_module(f"exp.{dataset_name}.{task}")
    experiment_class = get_experiment(task)
    if experiment_class is None:
        raise ValueError(f"Experiment '{task}' not found in registry")
    params = params or {}
    experiment: ExperimentBase = experiment_class(
        num_test=num_test,
        agent=agent,
        logs_dir=logs_dir,
        **params,
    )
    print(f"✅ Experiment class loaded: {experiment_class.__name__}")
    return experiment


def setup_experiment(config: Config) -> ExperimentBase:
    """Common setup logic for experiments: load agent and get experiment class."""
    # Load the agent instance
    agent = get_agent_instance(config.agent, config.model_name)

    # Get the experiment class
    experiment = get_experiment_class(
        dataset_name=config.dataset_name,
        task=config.task.name,
        num_test=config.num_test,
        agent=agent,
        params=config.task.params,
        logs_dir=config.logs_dir,
    )

    print(
        f"🚀 Running {config.task.name} of {config.dataset_name} on {config.agent.name} with {config.model_name} backbone for {config.num_test} experiments"
    )

    return experiment


def process_results(
    result_list: List[dict],
    config: Config,
    experiment: ExperimentBase,
    elapsed_time: Optional[float] = None,
    result_file_path: Optional[str] = None,
) -> None:
    """Common result processing: calculate metrics and save to JSON.
    
    Args:
        result_list: List of individual experiment results
        config: Experiment configuration
        experiment: Experiment instance
        elapsed_time: Optional elapsed time for the experiment
        result_file_path: Optional path to save results JSON, if provided, will overwrite result_list
    """
    if result_file_path is not None:
        # Load results from file for incremental finalization
        try:
            with open(result_file_path, "r") as f:
                result_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            result_list = []

        # Remove any existing metrics entry
        if (
            result_list
            and isinstance(result_list[-1], dict)
            and "metrics" in result_list[-1]
        ):
            result_list = result_list[:-1]

    try:
        metrics = experiment.calculate_metrics(result_list)
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        metrics = {"error": str(e), "error_type": type(e).__name__}

    result_list.append(
        {
            "metrics": metrics,
            "config": config.to_dict(),
            "elapsed_time": elapsed_time,
            "commit_hash": _get_git_commit_hash(),
        }
    )

    if result_file_path is not None:
        # Save back to the same file atomically
        result_path = Path(result_file_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file_path = result_path.with_suffix(".tmp")
        try:
            with open(temp_file_path, "w") as f:
                json.dump(result_list, f, indent=4)
            # Atomic replace
            os.replace(temp_file_path, result_path)
        except Exception:
            # Clean up temp file on error
            if temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
            raise
        logger.success(f"Results finalized and saved to {result_file_path}")
    else:
        save_results_as_json(result_list, config=config)


def build_overrides_from_args(
    task: Optional[str] = None,
    num_test: Optional[int] = None,
    model_name: Optional[ModelNames] = None,  # type: ignore
    dataset_name: Optional[str] = None,
    result_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    fix_test_cases_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build overrides dict from command line arguments."""
    overrides = {}
    if task is not None:
        overrides["task"] = task
    if model_name is not None:
        overrides["model_name"] = (
            model_name.value if hasattr(model_name, "value") else model_name
        )
    if dataset_name is not None:
        overrides["dataset_name"] = dataset_name
    if num_test is not None:
        overrides["num_test"] = int(num_test)
    if result_dir is not None:
        overrides["result_dir"] = result_dir
    if logs_dir is not None:
        overrides["logs_dir"] = logs_dir
    if fix_test_cases_dir is not None:
        overrides["fix_test_cases_dir"] = fix_test_cases_dir
    return overrides


def save_results_as_json(results, config: Config = None):
    """
    Save result list to "{result_dir}/{filename}.json".
    If config is provided, filename includes dataset_name, task, agent_name, timestamp.
    Otherwise, uses task and timestamp.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = config.result_dir if config is not None else "results"
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{config.dataset_name}_{config.task.name}_{config.agent.name}_{now}.json"
    )

    final_path = result_dir / filename
    temp_path = final_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(results, f, indent=4)
        # Atomic replace
        os.replace(temp_path, final_path)
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    logger.success(f"Results saved to {final_path}")
    return


async def run_concurrent_experiment(
    experiment: ExperimentBase,
    data_iter,
    n_jobs: int = 1,
    dry_run: bool = False,
    result_file_path: Optional[str] = None,
) -> List[dict]:
    """Run experiment with concurrent task execution and resume capability."""
    result_list = []

    semaphore = asyncio.Semaphore(n_jobs)
    file_lock = asyncio.Lock() if result_file_path else None
    testcase_idx = 0

    async def run_single(data, idx):
        async with semaphore:
            if dry_run:
                logger.info("Dry run mode: skipping actual experiment run.")
                result = {
                    "result": "dry_run",
                    "testcase_idx": data.get("testcase_idx", idx),
                    "query_id": str(uuid.uuid4()),
                }
            else:
                result = await experiment.run_agent(data)
                result["testcase_idx"] = data.get("testcase_idx", idx)
                experiment.cleanup(query_id=result.get("query_id"))
                await asyncio.sleep(0.2)

            # Save incrementally if result file path is provided
            if result_file_path and not dry_run and file_lock:
                async with file_lock:
                    save_result_incrementally(result, result_file_path)

            return result

    tasks = set()

    # Prime the queue with up to n_jobs tasks
    for _ in range(n_jobs):
        try:
            data = next(data_iter)
            tasks.add(asyncio.create_task(run_single(data, testcase_idx)))
            testcase_idx += 1
        except StopIteration:
            break

    while tasks:
        # Wait for any task to complete
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            result = await finished
            if result is not None:  # This should always be the case now
                result_list.append(result)
            # Schedule next task if data remains
            try:
                data = next(data_iter)
                tasks.add(asyncio.create_task(run_single(data, testcase_idx)))
                testcase_idx += 1
            except StopIteration:
                continue

    return result_list


async def run_experiment_with_resume(
    overrides: dict[str, str] = None,
    n_jobs: int = 1,
    dry_run: bool = False,
    config_path: str = "config.yaml",
    should_overwrite: bool = False,
    data_iterator_factory=None,
):
    """Run experiment with resume capability."""
    if overrides is None:
        overrides = {}

    config: Config = load_config(config_path, overrides)

    # Determine resume action
    resume_decision = determine_resume_action(config, should_overwrite)

    completed_testcase_indices = set()

    if resume_decision["action"] == "overwrite":
        logger.info("Overwrite requested, starting fresh...")
    elif resume_decision["action"] == "resume":
        result_file_path = resume_decision["resume_file"]
        logger.info(f"Resuming from: {result_file_path}")
        existing_results = load_existing_results(result_file_path)
        # Extract the set of testcase_idx that have already been completed
        completed_testcase_indices = {
            r.get("testcase_idx")
            for r in existing_results
            if r.get("testcase_idx") is not None
        }
        logger.info(
            f"Skipping {len(completed_testcase_indices)} already processed test cases: {sorted(completed_testcase_indices)}"
        )
    elif resume_decision["action"] == "complete_exists":
        logger.info(f"Complete results already exist: {resume_decision['resume_file']}")
        logger.info("Skipping experiment (use --overwrite to rerun)")
        return

    experiment = setup_experiment(config)

    if dry_run:
        logger.info("Dry run completed. No metrics to calculate.")
        return

    # Create result file path for incremental saving (always)
    result_file_path = None
    if resume_decision["action"] == "resume":
        result_file_path = resume_decision["resume_file"]
    elif resume_decision["action"] in ["fresh_start", "overwrite"]:
        # Create new result file path for incremental saving
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(config.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{config.dataset_name}_{config.task.name}_{config.agent.name}_{now}.json"
        )
        result_file_path = str(result_dir / filename)

    # NOTE: using fixed-version data_iterator
    start_time = time.time()
    data_iter = data_iterator_factory(config, completed_testcase_indices)
    await run_concurrent_experiment(
        experiment,
        data_iter,
        n_jobs=n_jobs,
        dry_run=dry_run,
        result_file_path=result_file_path,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        f"Experiment {config.dataset_name}/{config.task.name} used: {elapsed_time:.2f} seconds"
    )

    # Finalize results with metrics (always from file since we save incrementally)
    process_results(
        [],
        config,
        experiment,
        elapsed_time=elapsed_time,
        result_file_path=result_file_path,
    )


def _get_git_commit_hash() -> str:
    """Get the current git commit hash by reading .git files directly."""
    try:
        # Start from the current working directory and walk up to find .git
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:  # Stop at filesystem root
            git_dir = current_dir / ".git"
            if git_dir.exists() and git_dir.is_dir():
                break
            current_dir = current_dir.parent
        else:
            # If we reach here, no .git directory was found
            return ""

        head_file = git_dir / "HEAD"

        # Read the HEAD file to get the current branch reference
        with open(head_file, "r") as f:
            head_content = f.read().strip()

        if head_content.startswith("ref: "):
            # Extract the ref path (e.g., refs/heads/main)
            ref_path = head_content[5:]
            ref_file = git_dir / ref_path

            # Read the commit hash from the ref file
            with open(ref_file, "r") as f:
                return f.read().strip()
        else:
            # HEAD contains a commit hash directly (detached HEAD state)
            return head_content

    except (FileNotFoundError, IOError):
        # Return empty string if git repo not found or files unreadable
        return ""


def detect_resume_candidates(config: Config) -> List[Dict[str, Any]]:
    """Find existing result files that can be resumed for the current experiment config."""
    result_dir = Path(config.result_dir)
    if not result_dir.exists():
        return []

    pattern = f"{config.dataset_name}_{config.task.name}_{config.agent.name}_*.json"
    candidates = []

    for file_path in result_dir.glob(pattern):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if not data:
                continue

            # Check if this is a complete result file (has metrics entry at the end)
            final_entry = data[-1]
            has_metrics = isinstance(final_entry, dict) and "metrics" in final_entry

            candidates.append(
                {
                    "file_path": str(file_path),
                    "timestamp": file_path.stat().st_mtime,
                    "completed_cases": len(data) - 1 if has_metrics else len(data),
                    "has_metrics": has_metrics,
                    "config": final_entry.get("config", {}) if has_metrics else {},
                    "expected_cases": config.num_test,
                }
            )
        except (json.JSONDecodeError, IndexError, KeyError):
            continue

    # Sort by timestamp (most recent first)
    return sorted(candidates, key=lambda x: x["timestamp"], reverse=True)


def load_existing_results(result_file_path: str) -> List[dict]:
    """Load existing results from file for resume."""
    try:
        with open(result_file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def determine_resume_action(config: Config, should_overwrite: bool) -> Dict[str, Any]:
    """Determine whether to resume, overwrite, or start fresh based on existing files."""
    candidates = detect_resume_candidates(config)

    if not candidates:
        return {"action": "fresh_start", "resume_file": None}

    latest_candidate = candidates[0]

    if should_overwrite:
        return {"action": "overwrite", "resume_file": None}

    # Check if latest candidate is complete
    if latest_candidate["has_metrics"]:
        # Complete file exists - skip unless overwrite is requested
        return {
            "action": "complete_exists",
            "resume_file": latest_candidate["file_path"],
        }
    else:
        # Incomplete file exists - resume from it
        return {"action": "resume", "resume_file": latest_candidate["file_path"]}


def save_result_incrementally(result: dict, result_file_path: str) -> None:
    """Append a single result to the result file."""
    result_file_path = Path(result_file_path)

    # Read existing results
    if result_file_path.exists():
        try:
            with open(result_file_path, "r") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []
    else:
        existing_results = []

    # Remove metrics entry if present (will be recalculated at end)
    if (
        existing_results
        and isinstance(existing_results[-1], dict)
        and "metrics" in existing_results[-1]
    ):
        existing_results = existing_results[:-1]

    # Add new result
    existing_results.append(result)

    # Write to temporary file first, then atomically replace
    result_file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file_path = result_file_path.with_suffix(".tmp")
    try:
        with open(temp_file_path, "w") as f:
            json.dump(existing_results, f, indent=4)
        # Atomic replace
        os.replace(temp_file_path, result_file_path)
    except Exception:
        # Clean up temp file on error
        if temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        raise
