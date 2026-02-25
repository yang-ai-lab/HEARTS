import uuid
from textwrap import dedent
from typing import Any, Dict, List
import pandas as pd
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.parse_output import OutputStringParser

EXPERIMENT_NAME = "meal_react_comparison"

@register_experiment(EXPERIMENT_NAME)
class MealReactComparisonExperiment(CGMacrosExperiment):
    """
    Experiment: Given two 4-hour CGM windows (1 hour before and 3 hours after meal) from two subjects who ate similar meals (Carbs/Calories),
    one normal and one with prediabetes or diabetes, ask the agent to identify which CGM is from the normal subject.
    """
    PRED_KEY = "normal_subject"
    DEFAULT_FORECAST_MINUTES = 180  # 3 hours after meal
    DEFAULT_WINDOW_MINUTES = 60  # 1 hour before meal
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"
    MEAL_COLUMNS = ["Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber"]

    def __init__(
        self,
        task: str = EXPERIMENT_NAME,
        num_test: int = 50,
        logs_dir=None,
        agent=None,
        data_dir=None,
        forecast_minutes: int = DEFAULT_FORECAST_MINUTES,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        meal_similarity_threshold: float = 0.1,
    ):
        super().__init__(task, num_test, logs_dir, agent, data_dir)
        self.forecast_minutes = forecast_minutes
        self.window_minutes = window_minutes
        self.threshold = meal_similarity_threshold
        self.target_status = ["Prediabetes", "Diabetes"]

    def _get_pair_cgm_windows(self, target_meal, n_meal, i):
        # Helper to extract 4-hour CGM window for each meal
        def get_window(subject_id, meal_time):
            data = self.dataloader.load_data(subject_id)
            ts_col = pd.to_datetime(data[self.TIMESTAMP_COLUMN])
            meal_time = pd.to_datetime(meal_time)
            window_start = meal_time - pd.Timedelta(minutes=self.window_minutes)
            window_end = meal_time + pd.Timedelta(minutes=self.forecast_minutes)
            window_mask = (ts_col >= window_start) & (ts_col <= window_end)
            window_df = (
                data.loc[window_mask, [self.TIMESTAMP_COLUMN, self.CGM_COLUMN]]
                .dropna()
                .copy()
            )
            required_minutes = self.window_minutes + self.forecast_minutes
            if window_df.shape[0] < required_minutes:
                return None
            return window_df

        target_window = get_window(target_meal["subject_id"], target_meal["Timestamp"])
        n_window = get_window(n_meal["subject_id"], n_meal["Timestamp"])
        if target_window is None or n_window is None:
            return None
        # Deterministic order: half the test cases have target as A, half as B
        if i % 2 == 0:
            # Original order: A = target, B = normal
            pair = [("A", target_meal, target_window), ("B", n_meal, n_window)]
        else:
            # Swapped order: A = normal, B = target
            pair = [("A", n_meal, n_window), ("B", target_meal, target_window)]
        label_map = {p[0]: p[1]["diabetes_status"] for p in pair}
        gt = {k: v for k, v in label_map.items()}
        # For answer: which is target, which is normal
        return {
            "A": pair[0][2],
            "B": pair[1][2],
            "A_subject": pair[0][1]["subject_id"],
            "B_subject": pair[1][1]["subject_id"],
            "A_meal": pair[0][1],
            "B_meal": pair[1][1],
            "GT": gt,
            "Failures": 0,
        }

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = str(uuid.uuid4())
        # Save agent input: two CGM windows
        agent_input_data = {"A": data["A"], "B": data["B"]}
        self.save_data(agent_input_data, query_id=query_id)
        prompt = dedent(f"""\
        You are given two CGM data windows (A and B), each is a 4-hour window (1 hour before and 3 hours after a meal) from two different subjects. Both subjects ate a meal with similar carbohydrate and calorie content, but one subject is normal (non-diabetes) and the other has prediabetes or diabetes. Each CGM window is saved as 'input/A.csv' and 'input/B.csv', with columns: {self.TIMESTAMP_COLUMN} (datetime) and '{self.CGM_COLUMN}' (mg/dL).

        Your task: Based on the CGM data, decide which window (A or B) is from the normal subject and which is from the prediabetes or diabetes subject. Output your answer as a JSON object with the following format (no extra text):
        {{
            "{self.PRED_KEY}": "A"  # or "B"
        }}""")
        agent_output = await self.agent.query(
            prompt,
            agent_input_data,
            self.logs_dir,
            query_id=query_id,
        )
        solution, fail_reason = self.parse_output(agent_output)
        result = {
            "query_id": query_id,
            "A_subject": data["A_subject"],
            "B_subject": data["B_subject"],
            "A_meal": data["A_meal"],
            "B_meal": data["B_meal"],
            "GT": data["GT"],
            "solution": solution,
        }

        if fail_reason is not None:
            result["fail_reason"] = fail_reason

        return result

    def calculate_metrics(self, result_list: List[Dict[str, Any]]) -> ExperimentMetrics:
        if not result_list:
            return {"Accuracy": float("nan"), "Failures": 0}
        correct_count = 0
        failure_count = 0
        for result in result_list:
            solution = result["solution"]
            gt = result["GT"]
            if not solution:
                failure_count += 1
                continue
            pred = solution[self.PRED_KEY]
            if pred in gt and gt[pred] == "Normal":
                correct_count += 1
        accuracy = (
            correct_count / (len(result_list) - failure_count)
            if (len(result_list) - failure_count) > 0
            else float("nan")
        )
        print("Failures:", failure_count)
        print("Accuracy:", round(accuracy, 4))
        return {"Accuracy": round(accuracy, 4), "Failures": failure_count}

    def parse_output(self, content: str) -> tuple[Dict[str, Any], Any]:
        def validate(solution: dict):
            if solution.get(self.PRED_KEY) not in ["A", "B"]:
                raise ValueError(f"Invalid prediction: {solution.get(self.PRED_KEY)}")
            return

        return OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            expected_value_types={self.PRED_KEY: str},
            validator=validate,
        )
