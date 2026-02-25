import json
import os
import re
import zipfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display
from loguru import logger

from exp.utils.datatypes import ExperimentMetrics
from utils.agent import get_content_from_response
from utils.experiment_tags import (
    FrequencyGranularity,
    InputModality,
    InputSemanticDensity,
    Metrics,
    NumClasses,
    SequenceGranularity,
    TaskCategory,
    TemporalGranularity,
    get_all_datasets,
    get_all_tasks_for_dataset,
    get_experiment_tags,
)
from utils.save_log import load_json, read_state
from utils.str import extract_combine_xml_blocks

VALID_METRIC_KEYS = list(ExperimentMetrics.__annotations__.keys())

MetricLiteral = Literal[
    "MAE",
    "MSE",
    "MAPE",
    "SMAPE",
    "IoU",
    "Accuracy",
]


def write_item_to_jsonl(data, fp):
    with open(fp, "a") as f:
        json_line = json.dumps(data)
        f.write(json_line + "\n")
    return


def read_jsonl_file(fp, start: int = 0, end: int = -1):
    data = []
    with open(fp, "r") as f:
        for line_num, line in enumerate(f):
            if line_num >= start and (end == -1 or line_num < end):
                item = json.loads(line.strip())
                data.append(item)
    return data


def compress_file_to_zip(file_path: str, zip_path: str):
    """
    Compress a single file into a ZIP archive.

    Args:
        file_path (str): Path to the file to compress.
        zip_path (str): Path where the ZIP file will be created.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zipf:
        zipf.write(file_path, os.path.basename(file_path))


def uncompress_zip(zip_path: str, extract_to: str):
    """
    Extract all files from a ZIP archive to a specified directory.

    Args:
        zip_path (str): Path to the ZIP file to extract.
        extract_to (str): Directory where files will be extracted.
    """
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_to)


def _escape_xml_tags(text: str) -> str:
    """
    Escape XML tags by wrapping them in backticks for safe Markdown display.

    Args:
        text (str): Input text containing potential XML tags

    Returns:
        str: Text with XML tags wrapped in backticks
    """
    # Pattern to match simple XML tags like <tag> or </tag> (no spaces)
    xml_tag_pattern = r"</?\w+>"
    return re.sub(xml_tag_pattern, r"`\g<0>`", text)


def get_result_file_paths(result_dir: str, dataset: str, task: str):
    result_dir = Path(result_dir)
    pattern = f"{dataset}_{task}_*.json"
    file_paths = list(result_dir.glob(pattern))
    return file_paths


def load_experiment_results(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Separate predictions and metrics
    predictions = data[:-1]  # All but last item
    metrics_and_config = data[-1]  # Last item

    return {
        "predictions": predictions,
        "metrics": metrics_and_config.get("metrics", {}),
        "config": metrics_and_config.get("config", {}),
        "filename": os.path.basename(file_path),
    }


def read_experiment_log_states(logs_dir: str, task_name: str, query_id: str):
    """
    Read a specific log state (pickle file) from the logs directory.

    Args:
        logs_dir (str): Path to the logs directory containing task subdirectories with pickle files
        task_name (str): Name of the task
        query_id (str): Query ID (UUID) of the specific log state to load

    Returns:
        dict: The loaded log state containing messages, script, and other execution data, or None if not found
    """
    # Construct the expected pickle file path
    pickle_filename = f"{query_id}.pkl"
    pickle_path = os.path.join(logs_dir, task_name, pickle_filename)

    if not os.path.exists(pickle_path):
        logger.error(f"Log state file not found: {pickle_path}")
        return None

    try:
        log_state = read_state(pickle_path)
        logger.debug(f"Loaded log state: {pickle_path}")
        return log_state
    except Exception as e:
        logger.error(f"Error loading log state {pickle_path}: {e}")
        return None


def pretty_print_state_messages(state: dict, last_n: Optional[int] = None):
    """
    Pretty print state messages in a Jupyter notebook-friendly format.

    Args:
        state (dict): State dictionary containing 'messages' key with message objects
        last_n (Optional[int]): If provided, only print the last n messages
    """

    if last_n is not None:
        messages = state["messages"][-last_n:]
    else:
        messages = state["messages"]
    for i, m in enumerate(messages):
        try:
            # Add message separator for clarity
            # if i > 0:
            #     display(Markdown("---"))

            if hasattr(m, "type") and m.type == "ai":
                # AI message header
                display(Markdown("### 🤖 AI Message"))
                # if hasattr(m, "pretty_print"):
                # m.pretty_print()
                if hasattr(m, "content"):
                    # Fallback: display content directly
                    all_text = []
                    if isinstance(m.content, str):
                        # display(Markdown(_escape_xml_tags(m.content)))
                        all_text.append(_escape_xml_tags(m.content))
                    elif isinstance(m.content, list):
                        for c in m.content:
                            if isinstance(c, dict):
                                if "text" in c:
                                    all_text.append(_escape_xml_tags(c["text"]))
                                if "reasoning_content" in c and isinstance(
                                    c["reasoning_content"], dict
                                ):
                                    all_text.append(
                                        _escape_xml_tags(
                                            c["reasoning_content"].get("text", "")
                                        )
                                    )
                    display_content = "\n\n".join(all_text)
                    display_content = display_content.replace(
                        "`<execute>`", "```python\n"
                    ).replace("`</execute>`", "\n```")
                    display(Markdown(display_content))
            elif hasattr(m, "type") and m.type == "human":
                # Human message header
                display(Markdown("### 👤 Human Message"))
                if isinstance(m.content, list):
                    for c in m.content:
                        if isinstance(c, dict) and "image_url" in c:
                            # Display image
                            display(Image(url=c["image_url"]["url"]))
                        elif isinstance(c, dict) and "text" in c:
                            # Display text content
                            display(Markdown("```\n" + c["text"] + "\n```"))
                        else:
                            # Fallback for other content types
                            print(f"Content: {c}")
                elif hasattr(m, "content") and isinstance(m.content, str):
                    display(
                        Markdown("```\n" + _escape_xml_tags(str(m.content)) + "\n```")
                    )
                else:
                    # Fallback: display content directly
                    display(
                        Markdown("```\n" + _escape_xml_tags(str(m.content)) + "\n```")
                    )
            else:
                # Unknown message type
                display(
                    Markdown(f"### Unknown Message Type: {getattr(m, 'type', 'N/A')}")
                )
                if hasattr(m, "pretty_print"):
                    m.pretty_print()
                else:
                    print(f"Message: {m}")

        except Exception as e:
            # Error handling for malformed messages
            display(Markdown(f"**Error displaying message {i}:** {str(e)}"))
            print(f"Raw message: {m}")


# --- TOC: utils for trajectory evaluation ---
def is_result_json_valid(result_json_path: str) -> bool:
    data = load_json(result_json_path)
    if not data or not isinstance(data, list):
        return False
    last_entry = data[-1]
    if "metrics" not in last_entry or "config" not in last_entry:
        return False
    return True


def choose_lastest_valid_result_json_path(
    result_dir, dataset, task, should_check_valid=False
) -> Optional[str]:
    if isinstance(result_dir, str):
        result_dir = Path(result_dir)
    result_json_files = get_result_file_paths(result_dir, dataset, task)
    if len(result_json_files) == 0:
        raise ValueError(
            f"No result JSON files found for {dataset}-{task} in {result_dir}"
        )
    elif len(result_json_files) == 1:
        return str(result_json_files[0])
    else:
        # Multiple files found, choose the latest valid one
        valid_files = []
        for file_path in result_json_files:
            if not should_check_valid or is_result_json_valid(file_path):
                valid_files.append(file_path)
        if len(valid_files) == 0:
            raise ValueError(
                f"No valid result JSON files found for {dataset}-{task} in {result_dir}"
            )
        # Sort by modification time and return the latest
        valid_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(valid_files[0])


def get_dataset2task2result_fp(result_dir, should_check_valid=False):
    if isinstance(result_dir, str):
        result_dir = Path(result_dir)
    dataset2task2result_fp = defaultdict(dict)
    all_datasets = list(get_all_datasets())
    for dataset in all_datasets:
        tasks = list(get_all_tasks_for_dataset(dataset))
        dataset2task2result_fp[dataset] = {}
        for task in tasks:
            dataset2task2result_fp[dataset][task] = (
                choose_lastest_valid_result_json_path(
                    result_dir,
                    dataset,
                    task,
                    should_check_valid=should_check_valid,
                )
            )
    return dataset2task2result_fp


def from_result_entry_to_state_messages(result_entry: dict, last_entry: dict) -> list:
    """
    Convert a result entry containing predictions and metrics into a list of state messages.

    Args:
        result_entry (dict): Dictionary containing 'predictions' and 'metrics' keys

    Returns:
        list: List of message dictionaries representing the predictions and metrics
    """
    config = last_entry["config"]
    logs_dir = Path(config["logs_dir"])
    if "fail_reason" in result_entry:
        return None
    qid = result_entry["query_id"]
    state = read_state(logs_dir / config["task"]["name"] / f"{qid}.pkl")
    return state["messages"]


def from_messages_to_code_block_list(messages: list) -> list[str]:
    """
    Extract code blocks from a list of messages.

    Args:
        messages (list): List of message dictionaries
    Returns:
        list: List of code block strings extracted from the messages
    """
    out = []
    if not messages:
        return out
    for m in messages:
        if hasattr(m, "type") and m.type == "ai":
            content = get_content_from_response(m)
            if "<execute>" not in content:
                continue
            code_block = extract_combine_xml_blocks(content, tag="execute")
            out.append(code_block)
    return out


# --- TOC: utils for merge DF ---
"""NOTE:
concepts:
1. result_df: DataFrame containing experiment results with columns like dataset, task, model_id, metrics, and tags
2. agg_df: Aggregated DataFrame created by aggregating result_df by a specific tag column
"""


TAG_COL_NAME = Literal[
    "metric_type",
    "interm_cat",
    "n_class",
    "modality",
    "seq_gran",
    "tempo_gran",
    "freq_gran",
    "semantic_density",
]

COLNAME2TAG_ENUM: dict[TAG_COL_NAME, Enum] = {
    "metric_type": Metrics,
    "interm_cat": TaskCategory,
    "n_class": NumClasses,
    "modality": InputModality,
    "seq_gran": SequenceGranularity,
    "tempo_gran": TemporalGranularity,
    "freq_gran": FrequencyGranularity,
    "semantic_density": InputSemanticDensity,
}


def get_latest_result_file_path(result_dir: str | Path):
    # List all JSON files
    json_files = list(Path(result_dir).glob("*.json"))

    # Load and parse all JSON files

    # Load all results
    all_summaries = defaultdict(list)
    for json_file in json_files:
        try:
            # result = load_experiment_results(file_path)
            with open(json_file, "r") as f:
                data = json.load(f)
            summary = data[-1]  # Last item contains metrics and config
            if "config" not in summary:
                logger.warning(f"Skipping file without config or metrics: {json_file}")
                continue
            dataset_name = summary["config"]["dataset_name"]
            task_name = summary["config"]["task"]["name"]
            key = f"{dataset_name}-{task_name}"
            all_summaries[key].append((json_file, summary))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    dedup = {}
    for k in all_summaries:
        if len(all_summaries[k]) == 1:
            dedup[k] = all_summaries[k][0]
        else:
            # Sort by timestamp (lexicographically) and keep the latest
            sorted_results = sorted(
                all_summaries[k], key=lambda x: x[0].stat().st_mtime, reverse=True
            )
            if len(sorted_results) > 1:
                logger.debug("all_results filenames:")
                logger.debug([res[0] for res in sorted_results])
                logger.warning(
                    f"Multiple results found for {k}. Keeping the latest: {sorted_results[0][0]}"
                )
            dedup[k] = sorted_results[0]
    return dedup


def assemble_results_df(result_dir: str):
    """
    Assemble experiment results from JSON files in the specified directory into a pandas DataFrame.

    Args:
        result_dir (str): Path to the directory containing JSON result files

    Returns:
        pd.DataFrame: DataFrame containing experiment summaries with metrics
    """
    # List all JSON files
    json_files = list(Path(result_dir).glob("*.json"))

    # Load and parse all JSON files

    # Load all results
    all_results = []
    for json_file in json_files:
        try:
            # result = load_experiment_results(file_path)
            with open(json_file, "r") as f:
                data = json.load(f)
            result = data[-1]  # Last item contains metrics and config
            if "config" not in result:
                logger.warning(f"Skipping file without config or metrics: {json_file}")
                continue
            result["filename"] = json_file
            result["n_pred"] = len(data) - 1
            all_results.append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    # Deduplicate results: keep only the latest result for each dataset-task combination
    results_by_key = defaultdict(list)

    for result in all_results:
        dataset = result["config"]["dataset_name"]
        task = result["config"]["task"]["name"]
        key = f"{dataset}_{task}"
        results_by_key[key].append(result)

    # Keep only the latest result for each key (based on timestamp in filename)
    deduplicated_results = []
    for key, results in results_by_key.items():
        if len(results) == 1:
            deduplicated_results.append(results[0])
        else:
            # Sort by timestamp (lexicographically) and keep the latest
            sorted_results = sorted(
                results, key=lambda x: x["filename"].stat().st_mtime, reverse=True
            )
            if len(sorted_results) > 1:
                logger.debug("all_results filenames:")
                logger.debug([res["filename"] for res in sorted_results])
                logger.warning(
                    f"Multiple results found for {key}. Keeping the latest: {sorted_results[0]['filename']}"
                )
            deduplicated_results.append(sorted_results[0])

    # Parse experiment names and create summary dataframe
    experiment_summaries = []
    for result in deduplicated_results:
        dataset = result["config"]["dataset_name"]
        task = result["config"]["task"]["name"]
        metrics = result["metrics"]

        # Filter to valid metric keys only
        # filtered_metrics = {k: v for k, v in metrics.items() if k in VALID_METRIC_KEYS}
        filtered_metrics = {}
        for k in VALID_METRIC_KEYS:
            metric = metrics.get(k, None)
            if isinstance(metric, (int, float)):
                filtered_metrics[k] = metric
        # Handle MINMAX_SMAPE: use it as SMAPE if available
        if "MINMAX_SMAPE" in metrics:
            minmax_smape = metrics.get("MINMAX_SMAPE")
            if isinstance(minmax_smape, (int, float)):
                filtered_metrics["SMAPE"] = minmax_smape
        summary = {
            "dataset": dataset,
            "task": task,
            "filename": result["filename"],
            "num_predictions": result["n_pred"],
            **filtered_metrics,  # Include all valid metrics
        }
        experiment_summaries.append(summary)

    # Create DataFrame
    df_experiments = pd.DataFrame(experiment_summaries)
    df_experiments.sort_values(by=["dataset", "task"], inplace=True)

    return df_experiments


def merge_result_dfs(modelid2df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged_df = pd.concat(
        [df.assign(model_id=model_id) for model_id, df in modelid2df.items()],
        ignore_index=True,
    )
    merged_df.sort_values(by=["dataset", "task"], inplace=True)
    return merged_df


def get_specific_tag(dataset: str, task: str, tag_enum_type: Enum) -> list[Enum]:
    tags = get_experiment_tags(dataset, task)
    out = []
    for tag in tags:
        if isinstance(tag, tag_enum_type):
            out.append(tag)
    return out


def get_specific_tag_values(dataset: str, task: str, tag_enum_type: Enum) -> list[Enum]:
    tags = get_experiment_tags(dataset, task)
    out = []
    for tag in tags:
        if isinstance(tag, tag_enum_type):
            out.append(tag.value)
    return out


def add_tag_column(
    result_df: pd.DataFrame,
    column_name: TAG_COL_NAME,  # key of COLNAME2TAG_ENUM
) -> pd.DataFrame:
    result_df = result_df.copy()  # Work on a copy to avoid SettingWithCopyWarning
    result_df[column_name] = result_df.apply(
        lambda row: get_specific_tag_values(
            row["dataset"], row["task"], COLNAME2TAG_ENUM[column_name]
        ),
        axis=1,
    )
    return result_df


def aggregate_by_tag_column(
    result_df: pd.DataFrame,
    tag_column: TAG_COL_NAME,
    metrics_to_include: list[str] = None,
    explode_if_list: bool = True,
) -> pd.DataFrame:
    """
    Aggregate experiment results by a specified tag column.

    Args:
        df (pd.DataFrame): DataFrame containing experiment results with tag columns
        tag_column (str): Name of the tag column to aggregate by (e.g., 'interm_cat', 'n_class', 'modality')
        metrics_to_include (list[str], optional): List of metric names to include. Defaults to ['SMAPE', 'Accuracy', 'IoU']
        explode_if_list (bool, optional): Whether to explode the DataFrame if tag_column contains lists. Defaults to True

    Returns:
        pd.DataFrame: Aggregated DataFrame with tag-metric combinations as index and models as columns
    """
    if metrics_to_include is None:
        metrics_to_include = ["SMAPE", "Accuracy", "IoU"]

    # Work on a copy to avoid modifying original
    work_df = result_df.copy()

    # Ensure metric columns are numeric
    metric_cols = [
        "MAE",
        "MSE",
        "MAPE",
        "SMAPE",
        "Failures",
        "Accuracy",
        "IoU",
        "num_predictions",
    ]
    for col in metric_cols:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    # Explode if tag column contains lists (like modalities)
    if explode_if_list and tag_column in work_df.columns:
        # Check if any values are lists
        if work_df[tag_column].apply(lambda x: isinstance(x, list)).any():
            work_df = work_df.explode(tag_column)

    # Filter to only keep rows that share the same dataset & task with rows that model_id is "naive" and have valid metrics
    # Only apply this filtering if there are naive model rows present
    if (work_df["model_id"] == "naive").any():
        # First, identify dataset/task combinations that have valid "naive" model results
        naive_rows = work_df[
            (work_df["model_id"] == "naive")
            & work_df[metric_cols].notna().any(axis=1)  # At least one metric is not NaN
        ]
        valid_dataset_task_combinations = naive_rows[
            ["dataset", "task"]
        ].drop_duplicates()

        # Filter work_df to only include rows with matching dataset/task combinations
        work_df = work_df.merge(
            valid_dataset_task_combinations, on=["dataset", "task"], how="inner"
        )

    # Compute the number of experiments per tag value
    counts = count_experiments_by_tag(work_df, tag_column, is_list=False)

    # Group by model_id and tag column, aggregate numeric metrics with mean
    aggregated_df = (
        work_df.groupby(["model_id", tag_column]).mean(numeric_only=True).reset_index()
    )

    # Identify metric columns from the aggregated dataframe
    metric_columns = [
        col for col in aggregated_df.columns if col not in ["model_id", tag_column]
    ]

    # Melt to have metric as a column
    melted_df = aggregated_df.melt(
        id_vars=["model_id", tag_column],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )

    # Filter to only include specified metrics
    melted_df = melted_df[melted_df["metric"].isin(metrics_to_include)]

    # Create the new column names (tag_value + '-' + metric)
    melted_df["tag-metric"] = (
        melted_df[tag_column].astype(str) + "-" + melted_df["metric"]
    )

    # Pivot to wide format
    result_df = melted_df.pivot(index="tag-metric", columns="model_id", values="value")

    # Remove columns that are all NaN
    result_df = result_df.dropna(axis=0, how="all")

    # Sort the index for better readability
    # Define metric priority (higher is better metrics first)
    metric_priority = ["Accuracy", "IoU", "MAE", "SMAPE"]

    def metric_rank(idx):
        metric = idx.rsplit("-", 1)[-1]
        try:
            return metric_priority.index(metric)
        except ValueError:
            return len(metric_priority)

    # Within each metric group, sort rows by the mean across models
    # For metrics where larger is better (Accuracy, IoU) sort descending,
    # for metrics where smaller is better (MAE, SMAPE) sort ascending
    new_index = []
    for metric in metric_priority:
        # select rows that end with '-{metric}'
        group_rows = [idx for idx in result_df.index if idx.endswith(f"-{metric}")]
        if not group_rows:
            continue
        group_df = result_df.loc[group_rows]
        # compute mean across models, ignoring NaNs
        row_means = group_df.mean(axis=1, skipna=True)
        # choose sort order: descending for higher-is-better metrics, ascending for lower-is-better
        ascending = metric in ["MAE", "SMAPE"]  # smaller is better
        sorted_rows = row_means.sort_values(ascending=ascending).index.tolist()
        new_index.extend(sorted_rows)

    # keep any remaining rows (metrics not in metric_priority) at the end,
    # sorted by metric rank then name
    remaining = [idx for idx in result_df.index if idx not in new_index]
    remaining_sorted = sorted(remaining, key=lambda idx: (metric_rank(idx), idx))
    new_index.extend(remaining_sorted)

    result_df = result_df.reindex(new_index)

    if "naive" in result_df.columns:
        cols = result_df.columns.tolist()
        cols.remove("naive")
        cols = ["naive"] + cols
        result_df = result_df[cols]

    # Add the "num_exps" column by mapping the tag from the index to the counts
    result_df["num_exps"] = result_df.index.map(
        lambda idx: counts.get(idx.rsplit("-", 1)[0], 0)
    )

    # Sort the index based on tag order in the enum
    enum_class = COLNAME2TAG_ENUM[tag_column]
    tag_order = [member.value for member in enum_class]
    tag_to_index = {tag: idx for idx, tag in enumerate(tag_order)}

    def sort_key(idx):
        tag_metric = idx
        tag, metric = tag_metric.rsplit("-", 1)
        tag_idx = tag_to_index.get(tag, len(tag_order))  # Put unknown tags at the end
        metric_idx = (
            metric_priority.index(metric)
            if metric in metric_priority
            else len(metric_priority)
        )
        return (metric_idx, tag_idx)

    result_df = result_df.reindex(sorted(result_df.index, key=sort_key))

    return result_df


def get_diff_df_compared_naive(
    agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given an aggregated DataFrame with models including 'naive',
    compute the absolute difference of each model's metrics compared to 'naive'.

    Args:
        agg_df (pd.DataFrame): Aggregated DataFrame with models as columns and tag-metric as index

    Returns:
        pd.DataFrame: DataFrame with absolute differences compared to 'naive'
    """
    if "naive" not in agg_df.columns:
        raise ValueError("The input DataFrame must contain a 'naive' column.")

    diff_df = pd.DataFrame()
    columns_other_than_naive = [col for col in agg_df.columns if col != "naive"]
    for col in columns_other_than_naive:
        diff_df[col] = agg_df[col] - agg_df["naive"]
    diff_df["mean"] = diff_df[columns_other_than_naive].mean(axis=1)

    # Extract metric from index for grouping
    diff_df["metric"] = diff_df.index.str.rsplit("-", n=1).str[-1]

    # Define metric priority (same as in aggregate_by_tag_column)
    metric_priority = ["Accuracy", "IoU", "MAE", "SMAPE"]

    def metric_rank(metric):
        try:
            return metric_priority.index(metric)
        except ValueError:
            return len(metric_priority)

    diff_df["metric_rank"] = diff_df["metric"].apply(metric_rank)

    # Sort by metric rank (ascending), then by mean (descending)
    diff_df.sort_values(
        by=["metric_rank", "mean"], ascending=[True, False], inplace=True
    )

    # Drop temporary columns
    diff_df.drop(columns=["metric", "metric_rank"], inplace=True)

    return diff_df


def get_rel_diff_df_compared_naive(
    agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given an aggregated DataFrame with models including 'naive',
    compute the absolute difference of each model's metrics compared to 'naive'.

    Args:
        agg_df (pd.DataFrame): Aggregated DataFrame with models as columns and tag-metric as index

    Returns:
        pd.DataFrame: DataFrame with absolute differences compared to 'naive'
    """
    if "naive" not in agg_df.columns:
        raise ValueError("The input DataFrame must contain a 'naive' column.")

    diff_df = pd.DataFrame()
    columns_other_than_naive = [col for col in agg_df.columns if col != "naive"]
    for col in columns_other_than_naive:
        diff_df[col] = (agg_df[col] - agg_df["naive"]) / agg_df["naive"]
    diff_df["mean"] = diff_df[columns_other_than_naive].mean(axis=1)

    # Extract metric from index for grouping
    diff_df["metric"] = diff_df.index.str.rsplit("-", n=1).str[-1]

    # Define metric priority (same as in aggregate_by_tag_column)
    metric_priority = ["Accuracy", "IoU", "MAE", "SMAPE"]

    def metric_rank(metric):
        try:
            return metric_priority.index(metric)
        except ValueError:
            return len(metric_priority)

    diff_df["metric_rank"] = diff_df["metric"].apply(metric_rank)

    # Sort by metric rank (ascending), then by mean (descending)
    diff_df.sort_values(
        by=["metric_rank", "mean"], ascending=[True, False], inplace=True
    )

    # Drop temporary columns
    diff_df.drop(columns=["metric", "metric_rank"], inplace=True)

    return diff_df


def take_subset_experiments(
    result_df: pd.DataFrame,
    dataset_task_list: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Take a subset of experiments from the DataFrame based on dataset-task combinations.

    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        dataset_task_list (list[tuple[str, str]]): List of (dataset, task) tuples to filter by
    Returns:
        pd.DataFrame: Filtered DataFrame containing only the specified experiments
    """
    subset_df = result_df[
        result_df[["dataset", "task"]].apply(tuple, axis=1).isin(set(dataset_task_list))
    ]
    return subset_df


def merge_tags_in_result_df(
    result_df: pd.DataFrame,
    tag_col: TAG_COL_NAME,
    tags_to_merge: list[Enum | str],
    new_tag: str,
) -> pd.DataFrame:
    """
    Merge specified tags in the 'tag_col' column of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'freq_gran' column.
    tag_col (str): The name of the column containing the tags to be merged (e.g., 'freq_gran').
    tags_to_merge (list[Enum | str]): List of tags to be merged
    new_tag (str): The new tag to replace the merged tags (e.g., '<10Hz').

    Returns:
    pd.DataFrame: The DataFrame with merged tags in the 'freq_gran' column.
    """
    result_df = result_df.copy()
    tags_to_merge = [
        tag.value if isinstance(tag, Enum) else tag for tag in tags_to_merge
    ]
    result_df[tag_col] = result_df[tag_col].apply(
        lambda tags: [new_tag if tag in tags_to_merge else tag for tag in tags]
    )
    # Remove duplicates
    result_df[tag_col] = result_df[tag_col].apply(lambda tags: list(set(tags)))
    return result_df


def plot_metric_comparison(
    agg_df: pd.DataFrame,
    metric: MetricLiteral,
    desired_order=None,
    sort_column="qwen",
    title=None,
    tag_used=None,
):
    """
    General function to plot comparison of a specified metric across models, with naive as baseline if available.

    Args:
        agg_df (pd.DataFrame): Aggregated DataFrame with tag-metric as index and models as columns
        metric (str): The metric to plot (e.g., 'Accuracy', 'IoU')
        desired_order (list, optional): Specific order for the rows
        sort_column (str, optional): Column to sort by, defaults to 'qwen'
        tag_used (str, optional): Tag column used for labeling x-axis
    """
    # Filter rows where the index ends with the specified metric
    metric_rows = agg_df[agg_df.index.str.endswith(f"-{metric}")]

    # Sort the rows by the specified column in descending order
    metric_rows = metric_rows.sort_values(sort_column, ascending=False)

    # Remove the 'NA-{metric}' row if it exists
    na_row = f"NA-{metric}"
    if na_row in metric_rows.index:
        metric_rows = metric_rows.drop(na_row)

    # Enforce the specific order if provided
    if desired_order is not None:
        metric_rows = metric_rows.reindex(desired_order)

    # Initialize plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    handles = []
    labels = []

    # Plot the 'naive' column as a line if it exists
    if "naive" in metric_rows.columns:
        naive_handle = ax.plot(
            range(len(metric_rows)),
            metric_rows["naive"],
            marker="o",
            color="gray",
            alpha=0.5,
            label="naive",
        )
        handles.append(naive_handle[0])
        labels.append("naive")

    if title is None:
        if "naive" in metric_rows.columns:
            plt.title(f"{metric} Comparison Over Naive")
        else:
            plt.title(f"{metric} Comparison")
    else:
        plt.title(title)

    plt.ylabel(metric)
    if not tag_used:
        plt.xlabel("Tag")
    else:
        plt.xlabel(tag_used)
    plt.xticks(
        range(len(metric_rows)),
        metric_rows.index.str.replace(f"-{metric}", ""),
        rotation=45,
    )

    # Add points and lines for each model column except 'naive' if it exists
    columns_to_plot = [col for col in metric_rows.columns if col != "naive"]
    for col in columns_to_plot:
        scatter = ax.scatter(range(len(metric_rows)), metric_rows[col], s=50, alpha=0.5)
        ax.plot(range(len(metric_rows)), metric_rows[col], alpha=0.7)
        handles.append(scatter)
        labels.append(col.replace(f"-{metric}", ""))

    plt.legend(handles, labels)
    plt.tight_layout()
    plt.show()


def plot_relative_gain_comparison(
    agg_df: pd.DataFrame,
    metric: MetricLiteral,
    desired_order=None,
    sort_column="mean",
):
    """
    Plot relative gain comparison for a specified metric across models, showing the range of gains and the mean.

    Args:
        agg_df (pd.DataFrame): Difference DataFrame (e.g., from get_rel_diff_df_compared_naive) with tag-metric as index and models as columns
        metric (str): The metric to plot (e.g., 'Accuracy', 'IoU')
        desired_order (list, optional): Specific order for the rows
        sort_column (str, optional): Column to sort by, defaults to 'mean'
    """
    # Filter rows where the index ends with the specified metric
    metric_rows = agg_df[agg_df.index.str.endswith(f"-{metric}")]

    columns_other_than_naive_and_mean = [
        col for col in metric_rows.columns if col not in ["naive", "mean"]
    ]

    # calculate relative gain over naive
    for col in columns_other_than_naive_and_mean:
        metric_rows[col] = metric_rows[col] - metric_rows["naive"]
    metric_rows["mean"] = metric_rows[columns_other_than_naive_and_mean].mean(axis=1)

    # Sort the rows by the specified column in ascending order (since smaller gains might be first, but adjust as needed)
    metric_rows = metric_rows.sort_values(sort_column, ascending=True)

    # Remove the 'NA-{metric}' row if it exists
    na_row = f"NA-{metric}"
    if na_row in metric_rows.index:
        metric_rows = metric_rows.drop(na_row)

    # Enforce the specific order if provided
    if desired_order is not None:
        metric_rows = metric_rows.reindex(desired_order)

    # Compute min and max for the range of other models (excluding 'naive' and 'mean')
    min_other = metric_rows[columns_other_than_naive_and_mean].min(axis=1)
    max_other = metric_rows[columns_other_than_naive_and_mean].max(axis=1)

    # Plot the range as fill_between
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    fill_handle = ax.fill_between(
        range(len(metric_rows)),
        min_other,
        max_other,
        alpha=0.3,
        color="gray",
        label="Gain Range of Different Models",
    )

    # Plot the 'mean' column as a line
    mean_line = ax.plot(
        range(len(metric_rows)),
        metric_rows["mean"],
        color="black",
        alpha=1,
        label="Mean",
    )

    # Add scatter points for mean
    _ = ax.scatter(
        range(len(metric_rows)), metric_rows["mean"], color="black", s=50, alpha=0.7
    )

    plt.title(f"Relative Gain for {metric} Metrics")
    plt.ylabel("Rel Gain Over Naive")
    plt.xlabel("Tag-Metric")
    plt.xticks(range(len(metric_rows)), metric_rows.index, rotation=45)

    # Create legend
    handles = [mean_line[0], fill_handle]
    labels = ["Mean Gain Across Models", "Gain Range of Different Models"]
    plt.legend(handles, labels)
    plt.tight_layout()
    plt.show()


def cal_kappa(acc_obs, acc_exp):
    """difficulty normalization calculation using naive baseline as reference"""
    return (acc_obs - acc_exp) / (1 - acc_exp)


def convert_result_df_to_metric_df(df):
    """Convert a result DataFrame to a metric DataFrame by pivoting on dataset and task.
    This function takes a DataFrame containing results with columns for dataset, task,
    model_id, and metric, and transforms it into a pivoted DataFrame where each row
    represents a unique (dataset, task) pair, and each column represents a model_id
    with the corresponding metric value.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame with columns including 'dataset', 'task', 'model_id',
        and 'metric'. It should contain the results to be pivoted.
    Returns
    -------
    pandas.DataFrame
        A pivoted DataFrame with 'dataset' and 'task' as index columns, and
        'model_id' as column headers containing the metric values. In case of
        duplicate entries, the first value is used.
    """
    # Pivot the DataFrame so that each row is a unique (dataset, task) pair,
    # and columns are model_ids with metric values
    pivoted_df = (
        df.pivot_table(
            index=["dataset", "task"],
            columns="model_id",
            values="metric",
            aggfunc="first",  # In case of duplicates, take the first value
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    all_model_ids = df["model_id"].unique().tolist()
    if "naive" in all_model_ids:
        all_model_ids.remove("naive")
    pivoted_df["mean"] = pivoted_df[all_model_ids].mean(axis=1)
    return pivoted_df


def count_experiments_by_tag(
    result_df, tag_column: TAG_COL_NAME, is_list: bool = False
) -> pd.Series:
    """
    Count the number of unique experiments (dataset, task) for each value in the specified tag column.

    Args:
        result_df (pd.DataFrame): The result DataFrame containing experiment data.
        tag_column (str): The name of the tag column to count by (e.g., 'seq_gran').

    Returns:
        pd.Series: A series with tag values as index and counts as values.
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df_copy = result_df.copy()
    # Convert list values to tuples to make them hashable
    if is_list:
        result_df_copy[tag_column] = result_df_copy[tag_column].apply(tuple)
    unique_experiments = result_df_copy[
        ["dataset", "task", tag_column]
    ].drop_duplicates()
    return unique_experiments[tag_column].value_counts()


def aggregate_kappa_by_tag_column(
    kappa_metric_df: pd.DataFrame,
    tag_column: TAG_COL_NAME,
    explode_if_list: bool = True,
    model_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Aggregate kappa metric results by a specified tag column.

    Args:
        kappa_metric_df (pd.DataFrame): DataFrame containing kappa metric results with tag columns
        tag_column (str): Name of the tag column to aggregate by (e.g., 'interm_cat', 'n_class', 'modality')
        explode_if_list (bool, optional): Whether to explode the DataFrame if tag_column contains lists. Defaults to True

    Returns:
        pd.DataFrame: Aggregated DataFrame with tag as index and models as columns
    """
    # Work on a copy to avoid modifying original
    work_df = kappa_metric_df.copy()

    # Explode if tag column contains lists (like modalities)
    if explode_if_list and tag_column in work_df.columns:
        # Check if any values are lists
        if work_df[tag_column].apply(lambda x: isinstance(x, list)).any():
            work_df = work_df.explode(tag_column)

    # Identify model columns (assuming they are the numeric columns after 'task' and before tag columns)
    # From the structure, model columns are like 'deepseek', 'gpt_4_1_mini', etc.
    if model_cols is None:
        model_cols = [
            col
            for col in work_df.columns
            if col not in ["dataset", "task"] + list(COLNAME2TAG_ENUM.keys())
        ]

    # Compute the number of experiments per tag value
    counts = count_experiments_by_tag(work_df, tag_column, is_list=False)

    # Group by tag column, aggregate model columns with mean
    aggregated_df = work_df.groupby(tag_column)[model_cols].mean().reset_index()

    # Set tag_column as index
    agg_df = aggregated_df.set_index(tag_column)

    # std agg df
    std_agg_df = work_df.groupby(tag_column)["mean"].std().reset_index()
    std_agg_df = std_agg_df.set_index(tag_column)

    # Add the "num_exps" column by mapping the tag to the counts
    agg_df["num_exps"] = agg_df.index.map(lambda idx: counts.get(idx, 0))

    # Add 'mean' column as the mean across model columns
    agg_df["mean"] = agg_df[model_cols].mean(axis=1)
    agg_df["std"] = agg_df[model_cols].std(axis=1)
    agg_df["task_std"] = std_agg_df["mean"]

    # Sort the index based on tag order in the enum
    enum_class = COLNAME2TAG_ENUM[tag_column]
    tag_order = [member.value for member in enum_class]
    tag_to_index = {tag: idx for idx, tag in enumerate(tag_order)}

    agg_df = agg_df.reindex(
        sorted(agg_df.index, key=lambda idx: tag_to_index.get(idx, len(tag_order)))
    )

    return agg_df


def plot_box_plot_by_tag_from_metric_df(
    kappa_metric_df: pd.DataFrame,
    tag_column: TAG_COL_NAME,
    tag_column_name: str = None,
    full_order=None,
    ylabel: str = "Relative Improvement Score",
    title: str = None,
    should_plot_naive: bool = False,
):
    """
    Plot box plots of kappa metrics by tag from the aggregated DataFrame.
    """
    if not tag_column_name:
        tag_column_name = tag_column.replace("_", " ").title()

    # Explode the tag_column column to handle lists
    exploded_df = kappa_metric_df.explode(tag_column)
    exploded_df = exploded_df[exploded_df[tag_column] != "NA"]

    # Define the desired order based on the enum for the tag_column
    enum_class = COLNAME2TAG_ENUM[tag_column]
    if full_order is None:
        full_order = [member.value for member in enum_class]
    # Filter order to only include categories present in the data
    present_categories = exploded_df[tag_column].unique()
    order = [cat for cat in full_order if cat in present_categories]

    # Create the boxplot with the specified order and a nicer color palette
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=exploded_df, x=tag_column, y="mean", order=order, palette="Set3")
    if title is None:
        title = f"Distribution of Relative Improvement Scores by {tag_column_name}"
    plt.title(title)
    plt.xlabel(tag_column_name, labelpad=20)
    plt.ylabel(ylabel)
    
    if should_plot_naive:
        naive_series = kappa_metric_df.explode(tag_column).groupby(tag_column)['naive'].mean().reindex(order)
        plt.scatter(naive_series.index, naive_series, marker='o', color='blue', label='Naive Mean', zorder=10)
        plt.legend()

    # Calculate median values for each tag_column category
    medians = exploded_df.groupby(tag_column)["mean"].median()

    # Get y-axis limits to position annotations below the xlabels
    ymin, ymax = plt.ylim()
    offset = 0.05 * (ymax - ymin)  # Adjust offset as needed

    # Add text annotations for median values below the xlabels
    for i, cat in enumerate(order):
        plt.text(
            i,
            ymin - offset,  # Position below the xlabels
            f"{medians[cat]:.3f} (n={exploded_df[exploded_df[tag_column] == cat].shape[0]})",
            ha="center",
            va="top",
            fontsize=10,
        )

    plt.show()
    return
