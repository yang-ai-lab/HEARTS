import ast
from typing import Dict, List, Literal, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from scipy import stats

from .result_analysis import COLNAME2TAG_ENUM


class Hypothesis(TypedDict):
    """
    Definition of a statistical hypothesis test to run on benchmark results.

    Attributes:
        id: Unique identifier for this hypothesis (e.g., "h1", "agent_vs_naive").
        description: Human-readable description of what this hypothesis tests.
        test_type: Type of statistical test to perform:
            - "2indep": Compare two independent groups (Mann-Whitney U or Welch's T-test)
            - "1samp": Compare one group against a known value (Wilcoxon or 1-sample T-test)
            - "corr": Test correlation between two variables (Pearson or Spearman)
        groups: Tuple specifying the columns/values to compare:
            - For "2indep": (column_name_g1, column_name_g2) - two column names
            - For "1samp": (column_name, value_to_compare) - column name and scalar value
            - For "corr": (column_name_x, column_name_y) - two column names for correlation
        alternative: Direction of the test:
            - "two-sided": Test for difference/not equal (default)
            - "greater": Test if g1 > g2 or x > y
            - "less": Test if g1 < g2 or x < y
        filter: Optional dictionary of column:value pairs to filter the data before testing.
                Only rows where all filter conditions match will be included.
    """

    id: str
    description: str
    test_type: Literal["2indep", "1samp", "corr"]
    groups: Tuple[str, Union[str, float]]
    alternative: Literal["two-sided", "greater", "less"]
    filter: Union[Dict[str, str], None]


class HypothesisResult(TypedDict):
    id: str
    hypothesis: str
    test: str
    p_value: float
    stat: float
    supported: bool


class StatAnalysis:
    """
    Statistical analysis utilities for benchmark results.

    Provides methods for:
    - Loading and preprocessing benchmark result CSVs
    - Calculating gap closure metrics
    - Performing statistical tests with automatic normality-based test selection
    - Running configurable hypothesis test suites
    - Full analysis pipeline orchestration
    """

    @staticmethod
    def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
        """
        Load benchmark results CSV and preprocess columns for analysis.

        Handles:
        - Parsing string-encoded lists in result columns (e.g., "[0.85, 0.90]")
        - Extracting numeric values from n_class column for filtering/grouping
        - Creating primary_* columns with the first element of list columns

        Args:
            csv_path: Path to the CSV file containing merged benchmark results.

        Returns:
            DataFrame with:
            - Original columns with parsed list values
            - n_class_val: Numeric extracted value from n_class column
            - primary_{col}: First element of list columns (or None if not a list)
        """
        df = pd.read_csv(csv_path)

        list_cols = list(COLNAME2TAG_ENUM.keys())
        for col in list_cols:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        def parse_n_class(x):
            """
            Extract numeric value from n_class column for use in filtering/grouping.

            Handles various formats: [5], 5, "[5]", "[5.0]", etc.
            Returns NaN if parsing fails.
            """
            try:
                if isinstance(x, str):
                    val = ast.literal_eval(x)
                    if isinstance(val, list) and len(val) > 0:
                        return float(val[0])
                    elif isinstance(val, (int, float)):
                        return float(val)
                elif isinstance(x, (int, float)):
                    return float(x)
            except Exception:
                pass
            return np.nan

        df["n_class_val"] = df["n_class"].apply(parse_n_class)

        for col in list_cols:
            df[f"primary_{col}"] = df[col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
            )

        return df

    @staticmethod
    def calculate_gap_closure(
        df: pd.DataFrame, models: List[str], epsilon: float = 1e-9
    ) -> pd.DataFrame:
        """
        Calculate gap closure for each model relative to naive baseline.

        Gap closure measures how much of the performance gap between a perfect
        score (1.0) and the naive baseline has been closed by each model.

        Formula: gap_closure = (Model - Naive) / (1 - Naive)

        Interpretation:
        - 0.0: Same as naive baseline
        - 1.0: Achieved perfect score (1.0)
        - Negative: Worse than naive baseline
        - > 1.0: Exceeded perfect score (possible with some metrics)

        Args:
            df: DataFrame with naive and model columns containing scores.
            models: List of model column names to calculate gap closure for.
            epsilon: Small value to prevent division by zero when naive = 1.0.

        Returns:
            DataFrame with additional gap_{model} columns added.
        """
        df = df.copy()
        df["naive"] = pd.to_numeric(df["naive"], errors="coerce")

        for model in models:
            df[model] = pd.to_numeric(df[model], errors="coerce")
            mask = df["naive"].notna() & df[model].notna()
            df.loc[mask, f"gap_{model}"] = (
                df.loc[mask, model] - df.loc[mask, "naive"]
            ) / (1.0 - df.loc[mask, "naive"] + epsilon)

        return df

    @staticmethod
    def smart_test_2indep(
        g1: pd.Series, g2: pd.Series, alt: str = "two-sided"
    ) -> Tuple[str, float, float]:
        """
        Compare two independent groups using an appropriate statistical test.

        Automatically selects between Welch's T-test and Mann-Whitney U test
        based on the normality of both groups (Shapiro-Wilk test, alpha=0.05).

        Test selection:
        - If both groups are normally distributed: Welch's T-test (robust to unequal variances)
        - Otherwise: Mann-Whitney U test (non-parametric, rank-based)

        Args:
            g1: First group of values (e.g., agent scores).
            g2: Second group of values (e.g., baseline scores).
            alt: Alternative hypothesis:
                - "two-sided": Groups differ (g1 != g2)
                - "greater": g1 > g2
                - "less": g1 < g2

        Returns:
            Tuple of (test_name, p_value, statistic):
            - test_name: Name of the test performed
            - p_value: Two-sided p-value (for "two-sided" alternative)
            - statistic: Test statistic (t-statistic or U-statistic)
        """
        g1 = g1.dropna()
        g2 = g2.dropna()
        if len(g1) < 3 or len(g2) < 3:
            return "Insufficient Data", 1.0, 0.0

        try:
            _, p_n1 = stats.shapiro(g1)
            _, p_n2 = stats.shapiro(g2)
            is_normal = (p_n1 > 0.05) and (p_n2 > 0.05)
        except Exception:
            is_normal = False

        if is_normal:
            stat, p = stats.ttest_ind(g1, g2, equal_var=False, alternative=alt)
            test_name = "Welch's T-test"
        else:
            stat, p = stats.mannwhitneyu(g1, g2, alternative=alt)
            test_name = "Mann-Whitney U"

        return test_name, p, stat

    @staticmethod
    def smart_test_1samp(
        g: pd.Series, val: float, alt: str = "two-sided"
    ) -> Tuple[str, float, float]:
        """
        Compare a group of values against a known reference value.

        Automatically selects between one-sample T-test and Wilcoxon Signed-Rank
        test based on the normality of the difference from the reference value.

        Test selection:
        - If differences are normally distributed: One-sample T-test
        - Otherwise: Wilcoxon Signed-Rank test (non-parametric)

        Args:
            g: Series of values to test against the reference value.
            val: Reference value to compare against (e.g., 0.5 for chance level).
            alt: Alternative hypothesis:
                - "two-sided": Values differ from reference (g != val)
                - "greater": g > val
                - "less": g < val

        Returns:
            Tuple of (test_name, p_value, statistic):
            - test_name: Name of the test performed
            - p_value: Two-sided p-value
            - statistic: Test statistic (t-statistic or W-statistic)
        """
        g = g.dropna()
        if len(g) < 3:
            return "Insufficient Data", 1.0, 0.0

        try:
            _, p_n = stats.shapiro(g)
            is_normal = p_n > 0.05
        except Exception:
            is_normal = False

        if is_normal:
            stat, p = stats.ttest_1samp(g, val, alternative=alt)
            test_name = "1-sample T-test"
        else:
            stat, p = stats.wilcoxon(g - val, alternative=alt)
            test_name = "Wilcoxon Signed-Rank"

        return test_name, p, stat

    @staticmethod
    def smart_corr(x: pd.Series, y: pd.Series) -> Tuple[str, float, float]:
        """
        Test for correlation between two variables.

        Automatically selects between Pearson and Spearman correlation tests
        based on the normality of both variables.

        Test selection:
        - If both variables are normally distributed: Pearson correlation
        - Otherwise: Spearman rank correlation (monotonic relationships)

        Args:
            x: First variable values.
            y: Second variable values (must be same length as x).

        Returns:
            Tuple of (test_name, p_value, correlation):
            - test_name: Name of the test performed
            - p_value: P-value testing if correlation is significantly different from zero
            - correlation: Correlation coefficient (r for Pearson, rho for Spearman)
        """
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 5:
            return "Insufficient Data", 1.0, 0.0

        try:
            _, p_nx = stats.shapiro(x)
            _, p_ny = stats.shapiro(y)
            is_normal = (p_nx > 0.05) and (p_ny > 0.05)
        except Exception:
            is_normal = False

        if is_normal:
            corr, p = stats.pearsonr(x, y)
            test_name = "Pearson Corr"
        else:
            corr, p = stats.spearmanr(x, y)
            test_name = "Spearman Corr"

        return test_name, p, corr

    @staticmethod
    def run_h(h_id: str, desc: str, res: Tuple[str, float, float]) -> HypothesisResult:
        """
        Runs a statistical hypothesis test and returns the result as a dictionary.
        Args:
            h_id (str): The unique identifier for the hypothesis.
            desc (str): A description of the hypothesis being tested.
            res (Tuple[str, float, float]): A tuple containing the test name, p-value, and test statistic.
        Returns:
            HypothesisResult: A dictionary containing the hypothesis ID, description, test name, p-value, statistic, and a boolean indicating whether the hypothesis is supported (p < 0.05).
        """
        test_name, p, stat = res
        supported = p < 0.05
        return {
            "id": h_id,
            "hypothesis": desc,
            "test": test_name,
            "p_value": p,
            "stat": stat,
            "supported": supported,
        }


# run with `uv run -m utils.stat_analysis`
if __name__ == "__main__":
    df = StatAnalysis.load_and_preprocess_data("results/merged_metric_results.csv")

    agents = ["deepseek", "gpt_4_1_mini", "gpt_5_mini", "qwen", "qwen_coder", "mean"]
    df = StatAnalysis.calculate_gap_closure(df, agents)

    results = []

    def add_h(h_id: str, desc: str, res: Tuple[str, float, float]) -> None:
        results.append(StatAnalysis.run_h(h_id, desc, res))

    add_h(
        "H1",
        "GPT-5-mini outperforms GPT-4.1-mini (Gap Closure)",
        StatAnalysis.smart_test_2indep(
            df["gap_gpt_5_mini"], df["gap_gpt_4_1_mini"], "greater"
        ),
    )

    g_class = df[df["primary_interm_cat"] == "Physiology Classification"]["gap_mean"]
    g_fore = df[df["primary_interm_cat"] == "Forecasting"]["gap_mean"]
    add_h(
        "H2",
        "Physiology Classification is easier than Forecasting (Higher Gap)",
        StatAnalysis.smart_test_2indep(g_class, g_fore, "greater"),
    )

    add_h(
        "H3",
        "Average Agent Gap Closure > 0",
        StatAnalysis.smart_test_1samp(df["gap_mean"], 0, "greater"),
    )

    add_h(
        "H4",
        "DeepSeek performance correlates with GPT-5",
        StatAnalysis.smart_corr(df["gap_deepseek"], df["gap_gpt_5_mini"]),
    )

    g_ecg = df[df["primary_modality"] == "ECG"]["gap_mean"]
    g_eeg = df[df["primary_modality"] == "EEG"]["gap_mean"]
    add_h(
        "H5",
        "ECG tasks have higher gap than EEG tasks",
        StatAnalysis.smart_test_2indep(g_ecg, g_eeg, "greater"),
    )

    res_df = pd.DataFrame(results)
    print(res_df.to_string())
