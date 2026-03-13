"""
Experiment Metrics Data Types

This module defines the unified return type for all experiment calculate_metrics methods.
Use this TypedDict to standardize metric reporting across all experiments in HEARTS.

Usage:

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        # Your metric calculation logic here
        return {
            "MAE": 0.123,
            "Failures": 2,
            # ... other metrics as appropriate
        }
"""

from typing import Any, Dict, Union

from typing_extensions import TypedDict


class ExperimentMetrics(TypedDict, total=False):
    """
    Unified metrics return type for all experiment calculate_metrics methods.

    This TypedDict standardizes how experiments report their evaluation metrics.
    Since total=False, experiments only need to include the metrics they actually compute.

    REFACTORING GUIDE:
    When updating existing experiments to use this type:

    1. Import: from exp.utils.datatypes import ExperimentMetrics
    2. Update method signature: def calculate_metrics(...) -> ExperimentMetrics:
    3. Ensure return dict matches one of the patterns below
    4. Use 'additional_metrics' for experiment-specific metrics not covered here

    COMMON PATTERNS BY EXPERIMENT TYPE:

    REGRESSION EXPERIMENTS (bandpower_calculation, stat_calculation, forecasting):
    - Return MAE, MSE, MAPE, SMAPE, MINMAX_SMAPE as floats for single-target regression
    - For multi-target regression, use the single overall MAE/MSE/MAPE/SMAPE/MINMAX_SMAPE value and put detailed per-target metrics in additional_metrics
    - Always include 'Failures' count for invalid/missing predictions

    Example (single target):
        return {"MAE": 0.123, "MSE": 0.045, "MAPE": 15.6, "SMAPE": 12.1, "MINMAX_SMAPE": 0.089, "Failures": 2}

    Example (multi-target like stat_calculation):
        return {
            "MAE": 0.15,  # Overall average MAE across all targets
            "Failures": 1,
            "additional_metrics": {"MAE_per_target": {"target1": 0.1, "target2": 0.2}}
        }

    CLASSIFICATION EXPERIMENTS (activity_classification, disease_prediction):
    - Return 'Accuracy' (or 'Acc') as float between 0.0 and 1.0
    - Include 'Failures' for invalid predictions

    Example:
        return {"Accuracy": 0.856, "Failures": 3}

    FORECASTING/IMPUTATION EXPERIMENTS (meal_forecasting, signal_imputation):
    - Return MAE, MSE, MAPE, SMAPE, MINMAX_SMAPE for prediction quality
    - Use nested dicts for multi-dimensional predictions

    Example:
        return {"MAE": 0.089, "MSE": 0.034, "MAPE": 12.3, "SMAPE": 10.5, "MINMAX_SMAPE": 0.078, "Failures": 0}

    MIGRATION NOTES:
    - Existing experiments returning just floats: wrap in appropriate dict
    - Experiments with custom metrics: move to 'additional_metrics' dict
    - Ensure all numeric metrics are floats (not numpy types)
    - 'Failures' should always be int count of failed predictions
    """

    # Common regression metrics
    MAE: Union[float, int]
    """Mean Absolute Error - single float or int for overall MAE"""

    MSE: Union[float, int]
    """Mean Squared Error - single float or int for overall MSE"""

    MAPE: Union[float, int]
    """Mean Absolute Percentage Error - single float or int for overall MAPE"""

    SMAPE: Union[float, int]
    """Symmetric Mean Absolute Percentage Error - single float or int for overall SMAPE"""

    MINMAX_SMAPE: Union[float, int]
    """Min-Max Normalized Symmetric MAPE - scales SMAPE to [0, 1] range for comparison across different scales"""

    # Common classification metrics
    Accuracy: Union[float, int]
    """Classification accuracy as float between 0.0 and 1.0 (alternative to 'Acc')"""

    # Common localization metrics
    IoU: Union[float, int]
    """Intersection over Union - single float or int for overall IoU"""

    # Aggregated metrics (for multi-band/multi-variable cases)
    # Note: Use MAE directly instead of Average_MAE - MAE is now always a single float or int

    # Error tracking
    Failures: Union[int, Dict[str, int]]
    """Count of predictions or dict for per-target count that failed to parse or were invalid"""

    # Additional metrics that may appear in specific experiments
    additional_metrics: Dict[str, Any]
    """Any experiment-specific metrics not covered by standard fields above"""
