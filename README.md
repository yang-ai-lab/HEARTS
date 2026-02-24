# HEARTS

HEARTS is a comprehensive framework for benchmarking AI agents on various healthcare and physiological time-series tasks. It provides a standardized environment to evaluate agent performance across different datasets, tasks, and models.

## 🚀 Features

- **Modular Architecture**: separate components for experiments (`exp/`), agents (`agents/`), and utilities (`utils/`).
- **Diverse Datasets**: Support for multiple healthcare datasets (e.g., SHHS, Capture24, VitalDB, Bridge2AI Voice).
- **Variety of Tasks**: Includes Perception, Inference, Generation and Deduction tasks.
- **Agent Flexibility**: built-in support for different agent architectures (e.g., CodeAct).
- **Model Support**: Interfaces for major LLM providers (OpenAI, AWS Bedrock, Google Gemini, XAI).
- **Reproducibility**: "Frozen" experiment execution on fixed test cases to ensure consistent benchmarking.

## 🛠️ Installation

This project uses `uv` for dependency management.

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

## 🏃 Usage

The primary entry point for running experiments is `run_exp_freeze.py`. This script executes experiments on a set of pre-defined ("fixed") test cases to ensure reproducibility.

### Prerequisites

You must have a directory containing the fixed test cases (pickled data files). This directory is referred to as `fix_test_cases_dir`. The directory structure should follow: `fix_test_cases_dir/{dataset}/{task}/{index}.pkl`, where `{dataset}` is the dataset name, `{task}` is the task name, and `{index}` is an integer (e.g., `0.pkl`, `1.pkl`, `2.pkl`) corresponding to the test cases.

### Running an Experiment

You can run an experiment using the following command:

```bash
uv run run_exp_freeze.py --fix-test-cases-dir /path/to/test_cases
```

### Configuration

You can configure the experiment using `config.yaml` or command-line arguments.

**Example `config.yaml`:**

```yaml
dataset_name: shhs_remote
task:
  name: hypopnea_range
  params:
    example_param: value
num_test: 50
model_name: gpt-4.1-mini
agent:
  name: codeact
  params:
    verbose: true
fix_test_cases_dir: /path/to/fixed_cases  # Can also be set via CLI
result_dir: results/
```

**Command Line Overrides:**

CLI arguments override settings in `config.yaml`:

- `--task`: Task name
- `--dataset-name`: Dataset name
- `--num-test`: Number of test cases to run
- `--model-name`: Model identifier (e.g., `gpt-4.1`, `gemini-3-pro-preview`)
- `--result-dir`: Directory to save results
- `--logs-dir`: Directory to save agent logs
- `--n-jobs`: Number of concurrent jobs (default: 2)
- `--dry-run`: Run without calling the model (useful for checking setup)

## 📊 Results & Logs

Experiment results and agent execution logs are automatically saved.

### Results

Results are saved as JSON files in the directory specified by `result_dir` (default: `results/`).
The filename format is: `{dataset}_{task}_{agent}_{timestamp}.json`.

Each result file contains:
- A list of result entries for each test case (`query_id`, `subject_id`, `GT` (ground truth), `solution`).
- A final entry with calculated `metrics` (e.g., Accuracy) and the full experiment `config`.

### Logs (Agent State)

Agent execution states (including conversation history, code execution, thoughts) are saved as pickle files in `logs/{task}/{query_id}.pkl`.
These logs are crucial for debugging and analyzing agent behavior.

## 📂 Project Structure

```
.
├── agents/                 # Agent implementations
│   ├── base/               # Abstract base class for agents
│   ├── codeact/            # CodeAct agent implementation
├── exp/                    # Experiment definitions (Datasets & Tasks)
│   ├── base/               # Base experiment classes
│   ├── templates/          # Task templates (classification, forecasting, etc.)
│   ├── {dataset_name}/     # Dataset-specific experiment implementations
│   └── utils/              # Experiment registry and utilities
├── utils/                  # Core utilities
│   ├── exp.py              # Experiment runner logic
│   ├── model_enums.py      # Supported model definitions
│   ├── metric.py           # Evaluation metrics
│   └── ...
├── run_exp_freeze.py       # Main execution script for frozen experiments
├── config.yaml.example     # Example configuration file
└── pyproject.toml          # Project dependencies and metadata
```

## 🧪 Experiments

Experiments are defined in `exp/{dataset_name}/{task}.py`.
For details on adding new experiments, see [`exp/README.md`](exp/README.md).

**Examples of included datasets:**
- `bridge2ai_voice`: Voice analysis tasks (Parkinson's prediction, etc.)
- `capture24`: Activity tracking and health monitoring
- `coswara`: COVID-19 detection from audio
- `shhs_remote`: Sleep heart health study tasks
- `vitaldb`: Vital signs monitoring and prediction

## 🕵️ Agents

Agents are defined in `agents/{agent_name}/`.
For details on implementing new agents, see [`agents/README.md`](agents/README.md).

**Available Agents:**
- **CodeAct**: Uses executable code to interact with data and solve tasks.

## 🤝 Contributing

1.  Follow the instructions in `exp/README.md` to add new experiments.
2.  Follow the instructions in `agents/README.md` to add new agents.
3.  Ensure you run `uv sync` to keep dependencies up to date.