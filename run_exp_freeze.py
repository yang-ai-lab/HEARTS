import asyncio
import pickle
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from loguru import logger

from utils.exp import (
    Config,
    LogLevel,
    build_overrides_from_args,
    run_experiment_with_resume,
)
from utils.model_enums import ModelNames

load_dotenv()

app = typer.Typer()


def fix_data_iterator(config: Config, completed_testcase_indices: set = None):
    """Create data iterator that skips already processed test cases."""
    if completed_testcase_indices is None:
        completed_testcase_indices = set()

    fix_test_cases_dir = config.fix_test_cases_dir
    if fix_test_cases_dir is None:
        raise ValueError("config.fix_test_cases_dir must be set and not None.")
    fix_test_cases_dir = Path(fix_test_cases_dir)
    task_dir = fix_test_cases_dir / config.dataset_name / config.task.name
    if not task_dir.exists() or not task_dir.is_dir():
        logger.error(f"Task directory does not exist: {task_dir}")
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
    file_paths = list(task_dir.glob("*.pkl"))
    file_paths.sort(key=lambda x: int(x.stem))
    logger.info(f"Loading {len(file_paths)} fixed test cases from {task_dir}.")
    # NOTE: num_test depends on config
    for file_path in file_paths[: config.num_test]:
        idx = int(file_path.stem)  # Get index from filename
        # Skip already processed test cases based on specific indices
        if idx in completed_testcase_indices:
            logger.debug(f"Skipping already processed testcase_idx: {idx}")
            continue
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        # Ensure the data has the correct testcase_idx
        data["testcase_idx"] = idx
        yield data


@app.command()
def main(
    fix_test_cases_dir: str,  # NOTE: must have
    task: Optional[str] = None,
    num_test: Optional[int] = None,
    model_name: Optional[ModelNames] = None,  # type: ignore
    dataset_name: Optional[str] = None,
    result_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    n_jobs: int = 2,
    dry_run: bool = False,
    log_level: LogLevel = LogLevel.INFO,
    config_path: str = "config.yaml",
    should_overwrite: bool = False,
):
    logger.remove()
    logger.add(sys.stderr, level=log_level.value, enqueue=True)

    overrides = build_overrides_from_args(
        task=task,
        num_test=num_test,
        model_name=model_name,
        dataset_name=dataset_name,
        result_dir=result_dir,
        logs_dir=logs_dir,
        fix_test_cases_dir=fix_test_cases_dir,
    )

    asyncio.run(
        run_experiment_with_resume(
            overrides, n_jobs=n_jobs, dry_run=dry_run, config_path=config_path,
            should_overwrite=should_overwrite,
            data_iterator_factory=fix_data_iterator
        )
    )


if __name__ == "__main__":
    app()
