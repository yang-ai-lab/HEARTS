import uuid
from textwrap import dedent
from typing import Any, Dict, List
from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.parse_output import OutputStringParser

EXPERIMENT_NAME = "meal_img_classification"

@register_experiment(EXPERIMENT_NAME)
class MealImageClassificationExperiment(CGMacrosExperiment):
    """
    Experiment for classifying meal images from CGM data around a meal event.
    The agent is given CGM and meal info from the first 3 days, then a 4-hour CGM window around a randomly chosen meal (1 hour before and 3 hours after), and 4 choices of meal images, and is asked to choose which image is most likely this meal. The other 3 choices are from meals with more than 100 calories difference.
    """
    DEFAULT_FORECAST_MINUTES = 180  # 3 hours after meal
    DEFAULT_WINDOW_MINUTES = 60  # 1 hour before meal
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"
    MEAL_COLUMNS = [
        "Meal Type",
        "Calories",
        "Carbs",
        "Protein",
        "Fat",
        "Fiber",
        "Image path",
    ]
    PRED_KEY = "chosen_image"

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
        window_file = "cgm_window_around_meal"

        # Save agent input: reference (first 3 days CGM), and 4-hour CGM around meal
        agent_input_data = {window_file: data["window_df"]}
        agent_input_data.update(data["image_mapping"])
        self.save_data(agent_input_data, query_id=query_id)

        prompt = dedent(f"""\
        The CGM data for the 4 hours around the meal event (1 hour before and 3 hours after) is provided in 'input/{window_file}.csv'. There are two columns: {self.TIMESTAMP_COLUMN} (datetime) and '{self.CGM_COLUMN}' (CGM, mg/dL). A meal event occurred at {data["meal_time"]}.
        The 4 possible meal image options are saved as a.jpg, b.jpg, c.jpg, d.jpg in the input directory. So the paths are: input/a.jpg, input/b.jpg, input/c.jpg, input/d.jpg.
        Your task is to choose which image is most likely this meal based on the CGM data. Please output your final choice as a JSON object without any other text in the following format:
        {{
            "{self.PRED_KEY}": "a.jpg" # or "b.jpg", "c.jpg", "d.jpg"
        }}""")

        # Query the agent
        agent_output = await self.agent.query(
            prompt,
            agent_input_data,
            self.logs_dir,
            query_id=query_id,
        )
        solution, fail_reason = self.parse_output(agent_output)

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
            return {"Accuracy": float("nan"), "Failures": 0}
        correct_count = 0
        failure_count = 0
        for result in result_list:
            solution = result["solution"]
            gt = result["GT"]
            if not solution:
                failure_count += 1
                continue
            chosen = solution.get(self.PRED_KEY) if solution else None
            if chosen == gt:
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
        """
        Parse the agent's output to extract the chosen image path.
        Args:
            content: The response content from the agent.
        Returns:
            Dictionary with the parsed value.
        """
        return OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            expected_value_types={self.PRED_KEY: str},
        )
