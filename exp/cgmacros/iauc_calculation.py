import uuid
from textwrap import dedent
from typing import Any, Dict, List

from exp.cgmacros.base import CGMacrosExperiment
from exp.utils.datatypes import ExperimentMetrics
from exp.utils.registry import register_experiment
from utils.metric import Metrics
from utils.parse_output import OutputStringParser

EXPERIMENT_NAME = "iauc_calculation"

@register_experiment(EXPERIMENT_NAME)
class IaucCalculationExperiment(CGMacrosExperiment):
    """
    Experiment for calculating incremental Area Under the Curve (iAUC) for postprandial glucose response.
    The agent is provided with 2-hour CGM data starting from meal start and asked to calculate iAUC.
    The result is compared with ground truth iAUC calculated using calc_iauc.
    """

    PRED_KEY = "iauc"
    CGM_COLUMN = "Libre GL"
    TIMESTAMP_COLUMN = "Timestamp"

    def __init__(
        self,
        task: str = EXPERIMENT_NAME,
        num_test: int = 50,
        logs_dir=None,
        agent=None,
        data_dir=None,
    ):
        super().__init__(task, num_test, logs_dir, agent, data_dir)

    async def run_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = str(uuid.uuid4())
        cgm_file = "cgm_post_meal"

        agent_input_data = {cgm_file: data["cgm_df"]}
        self.save_data(agent_input_data, query_id=query_id)

        prompt = dedent(f"""\
        The CGM (continuous glucose monitoring) data for 2 hours after a meal is provided in 'input/{cgm_file}.csv'. There are two columns: 'Time (min)' and 'CGM (mg/dL)'.
        Please calculate the incremental Area Under the Curve (iAUC) for the postprandial glucose response. Use first CGM value as baseline. Please output your calculated iAUC value as a JSON object without any other text in the following format:
        {{
            "{self.PRED_KEY}": [number]
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
            return {
                "MAE": float("nan"),
                "MAPE": float("nan"),
                "SMAPE": float("nan"),
                "Failures": 0,
            }  # use NaN instead of 0
        preds = []
        gts = []
        failure_count = 0
        smape_invalid_count = 0
        for result in result_list:
            pred = (
                result["solution"].get(self.PRED_KEY, None)
                if result["solution"]
                else None
            )
            gt = result["GT"]
            if pred is not None and isinstance(pred, (int, float)):
                preds.append(pred)
                gts.append(gt)
            else:
                failure_count += 1
        if preds:
            mae = Metrics.mae({"iauc": preds}, {"iauc": gts})["iauc"]
            mape = Metrics.mape({"iauc": preds}, {"iauc": gts})["iauc"]
            smape_result, invalid_count = Metrics.smape(
                {"iauc": preds}, {"iauc": gts}
            )
            smape = smape_result["iauc"]
            smape_invalid_count += invalid_count
        else:
            mae = float("nan")
            mape = float("nan")
            smape = float("nan")
        print("Failures:", failure_count)
        print("MAE:", round(mae, 4))
        print("MAPE:", round(mape, 4))
        print("SMAPE:", round(smape, 4))
        return {
            "MAE": round(mae, 4),
            "MAPE": round(mape, 4),
            "SMAPE": round(smape, 4),
            "Failures": failure_count,
            "additional_metrics": {
                "smape_invalid_count": smape_invalid_count
            },
        }

    def parse_output(self, content: str) -> tuple[Dict[str, Any], Any]:
        """
        Parse the agent's output to extract the predicted iAUC value.
        Args:
            content: The response content from the agent.
        Returns:
            Dictionary with the parsed value.
        """
        return OutputStringParser.parse_dict(
            content,
            expected_keys=[self.PRED_KEY],
            value_converter=float,
        )
