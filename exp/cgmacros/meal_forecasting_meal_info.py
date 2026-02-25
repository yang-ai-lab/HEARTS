import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List
import numpy as np
from loguru import logger
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

EXPERIMENT_NAME = "meal_forecasting_meal_info"

@register_experiment(EXPERIMENT_NAME)
class MealForecastingMealInfoExperiment(CGMacrosExperiment):
    """
    Experiment for forecasting CGM values after a meal event.
    The agent is given all CGM and meal info from the first three days, then a 1-hour CGM window before a randomly chosen meal, and is asked to forecast CGM for the next 30 minutes after the meal starts.
    """

    DEFAULT_FORECAST_MINUTES = 30
    DEFAULT_WINDOW_MINUTES = 60  # 1hr before meal as reference
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"
    MEAL_COLUMNS = ["Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber"]
    PRED_KEY = "forecasted_cgm"

    def __init__(
        self,
        task: str = EXPERIMENT_NAME,
        num_test: int = 50,
        logs_dir=None,
        agent=None,
        data_dir=None,
        forecast_minutes: int = DEFAULT_FORECAST_MINUTES,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
    ):
        super().__init__(task, num_test, logs_dir, agent, data_dir)
        self.forecast_minutes = forecast_minutes
        self.window_minutes = window_minutes

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = str(uuid.uuid4())
        cgm_file = "cgm_first_three_days"
        meal_file = "meal_info_first_three_days"
        window_file = "cgm_window_before_meal"

        # Save agent input: reference (first three days CGM & meal info), and 1-hour CGM before meal
        agent_input_data = {
            cgm_file: data["reference_cgm_df"],
            meal_file: data["reference_meal_info_df"],
            window_file: data["window_df"],
        }
        self.save_data(agent_input_data, query_id=query_id)

        meal_info_str = ", ".join([f"{k}: {v}" for k, v in data["meal_info"].items()])
        prompt = dedent(f"""\
        The CGM (continuous glucose monitoring) data for this subject's first three days is provided in 'input/{cgm_file}.csv'. There are two columns: Timestamp (datetime) and 'Libre GL' (CGM, mg/dL).
        All meal events for the first three days are provided in 'input/{meal_file}.csv', with columns: Timestamp, Meal Type, Calories, Carbs, Protein, Fat, Fiber. These meal events are for your reference.
        The CGM data for the 1 hour before the following meal event is provided in 'input/{window_file}.csv'.
        A meal event occurred at {data["meal_time"]}. The information for this meal is: {meal_info_str}.
        Please forecast the CGM values for the next {self.forecast_minutes} minutes (in mg/dL) after the meal starts. Save your forecasted values in order as a list in json file at `output/solution.json`. Make sure there are {self.forecast_minutes} values. When finished, put your methodology to solve this task as your answer.""")

        _ = await self.agent.query(
            prompt,
            agent_input_data,
            self.logs_dir,
            query_id=query_id,
        )

        # parse output file here
        solution, fail_reason = self.parse_output(query_id=query_id)

        result = {
            "query_id": query_id,
            "subject_id": data["subject_id"],
            "meal_time": str(data["meal_time"]),
            "GT": data["GT"],
            "solution": solution,
        }

        if fail_reason is not None:
            result["fail_reason"] = fail_reason

        return result

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        if not result_list:
            # Fix: When result_list is empty, no tests were attempted, so failures should be 0, not self.num_test
            # This prevents incorrectly reporting all intended tests as failures when no data was processed
            return {
                "MAE": 0.0,
                "MSE": 0.0,
                "MAPE": 0.0,
                "Failures": 0,
            }
        # Calculate per-sample metrics and average them for accurate evaluation
        # This fixes the bug where concatenating all predictions treated them as a single long time series
        mae_values = []
        mse_values = []
        mape_values = []
        smape_values = []
        minmax_smape_values = []  # for minmax SMAPE
        failure_count = 0
        smape_invalid_count = 0
        minmax_smape_invalid_count = 0  # for minmax SMAPE
        for result in result_list:
            pred = result["solution"]
            gt = result["GT"]
            if pred and gt:
                pred = pred[: self.forecast_minutes]
                # Calculate metrics for this single test case/sample
                sample_mae = Metrics.mae({self.PRED_KEY: pred}, {self.PRED_KEY: gt})[
                    self.PRED_KEY
                ]
                sample_mse = Metrics.mse({self.PRED_KEY: pred}, {self.PRED_KEY: gt})[
                    self.PRED_KEY
                ]
                sample_mape = Metrics.mape({self.PRED_KEY: pred}, {self.PRED_KEY: gt})[
                    self.PRED_KEY
                ]
                # Calculate SMAPE for this sample
                smape_result, invalid_count = Metrics.smape(
                    {self.PRED_KEY: pred}, {self.PRED_KEY: gt}
                )
                sample_smape = smape_result[self.PRED_KEY]
                smape_invalid_count += invalid_count

                # Calculate MINMAX_SMAPE for this sample
                pred_array = np.array(pred)
                gt_array = np.array(gt)
                pred_min = np.min(pred_array)
                pred_max = np.max(pred_array)
                gt_min = np.min(gt_array)
                gt_max = np.max(gt_array)
                common_min = min(pred_min, gt_min)
                common_max = max(pred_max, gt_max)
                if common_max - common_min > 0:
                    pred_array_scaled = (pred_array - common_min) / (
                        common_max - common_min
                    )
                    gt_array_scaled = (gt_array - common_min) / (
                        common_max - common_min
                    )
                    if np.all(gt_array_scaled == 0):
                        minmax_smape_result, minmax_invalid_count = Metrics.smape(
                            {self.PRED_KEY: pred_array}, {self.PRED_KEY: gt_array}
                        )
                    else:
                        minmax_smape_result, minmax_invalid_count = Metrics.smape(
                            {self.PRED_KEY: pred_array_scaled.tolist()},
                            {self.PRED_KEY: gt_array_scaled.tolist()},
                        )
                    sample_minmax_smape = minmax_smape_result[self.PRED_KEY]
                    minmax_smape_invalid_count += minmax_invalid_count
                    if sample_minmax_smape is not None and not np.isnan(
                        sample_minmax_smape
                    ):
                        minmax_smape_values.append(sample_minmax_smape)
                else:
                    minmax_smape_invalid_count += 1

                if sample_mae is not None and not np.isnan(sample_mae):
                    mae_values.append(sample_mae)
                if sample_mse is not None and not np.isnan(sample_mse):
                    mse_values.append(sample_mse)
                if sample_mape is not None and not np.isnan(sample_mape):
                    mape_values.append(sample_mape)
                if sample_smape is not None and not np.isnan(sample_smape):
                    smape_values.append(sample_smape)
            else:
                failure_count += 1
        # Average the per-sample metrics across all successful test cases
        avg_mae = np.mean(mae_values) if mae_values else float("nan")
        avg_mse = np.mean(mse_values) if mse_values else float("nan")
        avg_mape = np.mean(mape_values) if mape_values else float("nan")
        avg_smape = np.mean(smape_values) if smape_values else float("nan")
        avg_minmax_smape = (
            np.mean(minmax_smape_values) if minmax_smape_values else float("nan")
        )
        print("Failures:", failure_count)
        print("MAE:", round(avg_mae, 4))
        print("MSE:", round(avg_mse, 4))
        print("MAPE:", round(avg_mape, 4))
        print("SMAPE:", round(avg_smape, 4))
        print("MINMAX_SMAPE:", round(avg_minmax_smape, 4))
        return {
            "MAE": round(avg_mae, 4),
            "MSE": round(avg_mse, 4),
            "MAPE": round(avg_mape, 4),
            "SMAPE": round(avg_smape, 4),
            "MINMAX_SMAPE": round(avg_minmax_smape, 4),
            "Failures": failure_count,
            "additional_metrics": {
                "smape_invalid_count": smape_invalid_count,
                "minmax_smape_invalid_count": minmax_smape_invalid_count,
            },
        }

    def parse_output(self, query_id: str):
        try:
            with open(
                Path("agent_working") / query_id / "output" / "solution.json", "r"
            ) as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read output file for {query_id}: {e}")
            return [], str(e)

        def validator(solution: List[Any]) -> None:
            if len(solution) < self.forecast_minutes:
                raise ValueError(
                    f"Expected at least {self.forecast_minutes} forecasted values, got {len(solution)}"
                )
            for value in solution:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Forecasted value {value} is not a number")

        out, fail_reason = OutputStringParser.parse_list(
            content=content, validator=validator
        )
        return out, fail_reason
