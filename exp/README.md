# Experiment Structure and Naming Conventions

This directory contains experiments for HEARTS. Experiments are organized by dataset and task, following a consistent naming and structure paradigm.

## Directory Structure

```
exp/
├── base/
│   └── base.py          # Abstract base class for all experiments
├── utils/
│   └── registry.py      # Registry for experiments and agents
├── {dataset_name}/      # Dataset-specific directory
│   ├── base.py          # Dataset-specific base class (optional)
│   ├── {task}.py        # Experiment implementation
│   └── utils.py         # Dataset-specific utilities (optional)
```

## Registry System

Experiments and agents are registered using decorators from `exp/utils/registry.py` to enable dynamic loading and retrieval.

To register an experiment:

```python
from exp.utils.registry import register_experiment

@register_experiment("task_name")
class TaskNameExperiment(ExperimentBase):
    ...
```

The experiment class can then be retrieved using `get_experiment("task_name")`.

## Naming Conventions

### Task Names

- Use lowercase with underscores (snake_case)
- Examples: `bandpower`, `eye_blink`, `hypopnea_range`

### Class Names

- Convert task name to PascalCase and append "Experiment"
- Formula: `''.join(word.capitalize() for word in task.split('_')) + 'Experiment'`
- Examples:
  - `bandpower` → `BandpowerExperiment`
  - `eye_blink` → `EyeBlinkExperiment`
  - `hypopnea_range` → `HypopneaRangeExperiment`

### File Names

- Located in `exp/{dataset_name}/{task}.py`
- Examples:
  - Dataset: `shhs_local`, Task: `hypopnea_range` → `exp/shhs_local/hypopnea_range.py`

## Creating a New Experiment

1. **Choose dataset and task names** following the conventions above
2. **Create the directory** if it doesn't exist: `exp/{dataset_name}/`
3. **Create the file** `exp/{dataset_name}/{task}.py`
4. **Implement the class** inheriting from `ExperimentBase` or a dataset-specific base class (e.g., `SHHSExperiment` for SHHS dataset)
5. **Register the experiment** using the `@register_experiment('{task}')` decorator from `exp.utils.registry`
6. **Implement all abstract methods**:
   - `prepare_data()`: Preprocessing logic (setup dataloader, etc.)
   - `data_iterator()`: Generator yielding data dicts for each test, typically including:
     - ground truth for later evaluation
     - data agent can access
     - other relevant info want to save in `result`
   - `run_agent(data)`: Async method to run agent on single data dict, typically includes:
     - separate ground truth & data for agent
     - save agent input data at specific location using `save_data`
     - write a prompt for agent, query agent
     - parse agent output
     - assemble and return `result` dict
   - `parse_output(content, query_id)`: Parse agent output into solution dict, used in `run_agent`. Try to use `utils.parse_output.OutputStringParser` for common parsing logic.
   - `calculate_metrics(result_list)`: Calculate and return metrics (general metrics are implemented at `utils.metric.Metrics`)
   - `save_data(data, query_id)`: Save input data for agent at specified location that it can access.
   - `cleanup(query_id)`: Cleanup any temporary files or directories used during the experiment.

## Base Class: ExperimentBase

All experiments inherit from `ExperimentBase` which provides:

- Initialization with `num_test`, `logs_dir`, and `agent`
- Automatic logs directory creation
- Abstract methods that must be implemented

### Dataset-Specific Base Classes

Some datasets may have specific base classes that extend `ExperimentBase` with dataset-specific functionality. For example, `SHHSExperiment` in `exp/shhs_local/base.py` provides SHHS-specific data handling.

## Configuration

Experiments are configured via `config.yaml` in the root directory:

- `dataset_name`: The dataset folder name (e.g., "shhs_local")
- `task`: The task name (e.g., "hypopnea_range")
- `num_test`: Number of test iterations
- `model_name`: OpenAI model to use
- `agent`: Configuration for the agent
  - `name`: Agent name (e.g., "codeact")
  - `params`: Additional parameters for the agent
- `result_dir`: Directory to save results (default: `results/`)
- `logs_dir`: Directory to save logs (default: `logs/`)

## Running Experiments

Use `run_exp_freeze.py` to execute experiments. This project uses `uv` for dependency management.

```bash
uv run python run_exp_freeze.py
```

It automatically:

1. Loads configuration from `config.yaml`
2. Retrieves the experiment class from the registry using the task name
3. Runs the experiment and saves results
4. Calculates metrics

Example config for hypopnea_range on SHHS dataset:

```yaml
dataset_name: shhs_local
task: hypopnea_range
num_test: 50
model_name: gpt-4o-mini
agent:
  name: codeact
  params:
    verbose: true
```

## Results and Logs

### Result JSON Files
Experiment results are saved as JSON files in `results/` (or configured `result_dir`) with the pattern `{dataset}_{task}_{agent}_{timestamp}.json`.
- Individual result entries contain `query_id`, `subject_id`, `GT` (ground truth), and `solution` (parsed agent response).
- Final entry contains `metrics` and `config`.

### Logs
Agent execution states are saved as pickle files in `logs/{task}/{query_id}.pkl`. These contain the full conversation history, code execution details, and agent state for debugging.

# Development

After you made changes to an experiment, please update `exp/changelog.md` with a brief description of the changes made for tracking purposes.

Ensure your code is formatted and linted:
```bash
uv run ruff format .
uv run ruff check . --fix
```
