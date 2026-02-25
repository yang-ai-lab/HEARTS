from pathlib import Path
from typing import Any, Optional, Type, TypeVar
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from utils.str import extract_combine_xml_blocks
from utils.str_sanitize import extract_md_img, md_img_to_url
from utils.agent import get_content_from_response, JupyterKernel, JupyterLocalKernel, convert_reasoning_to_text


class CodeActState(MessagesState):
    """State for CodeAct agent."""

    script: Optional[str]
    """The Python code script to be executed."""
    context: dict[str, Any]
    """Dictionary containing the execution context with available tools and variables."""
    count_thoughts: int = 0
    """Counter for number of responses only with thoughts."""
    final_answer: Optional[str] = None
    """The final answer provided by the agent."""


StateSchema = TypeVar("StateSchema", bound=CodeActState)
StateSchemaType = Type[StateSchema]
console = Console()


def create_default_prompt(base_prompt: Optional[str] = None):
    """Create default prompt for the CodeAct agent."""
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """\
You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will be using an interactive coding environment to assist you throughout the process.

At each turn, you should first provide your step-by-step thinking for solving the task. Your thought process should be enclosed using "<thought>" tag, for example: <thought> I need to print "Hello World!" </thought>.

After that, you have two options:

1) Interact with a Python programming environment and receive the corresponding output. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.
2) Directly provide a solution that adheres to the required format for the given task. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>.
"""

    prompt += """
Variables defined at the top level of previous code snippets can be referenced in your code."""

    prompt += """
Due to the limitations of the context length, you should avoid printing the whole sequence of any given data during your thought process or code execution."""

    prompt += """
--- 

Task:
One heart rate (HR) signal is saved in the file 'input/HR.npy', which has sampling frequency at 1 Hz, unit is beat per minute (BPM). Calculate the time (in seconds) when the HR are within normal resting range (60 - 100 BPM). Please output your final answer in the following JSON format without any other text:
\{
    "time_in_range": [float, total time in seconds when HR is within normal resting HR range],
\}

Assistant:
<thought> To solve this task, I will first load the HR signal from the provided .npy file. Then, I will analyze the signal to identify the time points where the HR values fall within the normal resting range of 60 to 100 BPM. Finally, I will calculate the total time in seconds that the HR is within this range and format the result in the specified JSON format. </thought>
<execute>
import numpy as np

HR = np.load('input/HR.npy')
print(f"Loaded HR signal with {len(HR)} data points.")
</execute>

Observation:
Loaded HR signal with 300 data points.

Assistant:
<thought> Now that I have loaded the HR signal, I will proceed to identify the time points where the HR values are between 60 and 100 BPM. I will then calculate the total time in seconds that the HR is within this range. </thought>
<execute>
time_in_range = np.sum((HR >= 60) & (HR <= 100))
print(f"Calculated time in range: {time_in_range} seconds.")
</execute>

Observation:
Calculated time in range: 240 seconds.

Assistant:
<thought> I have calculated the total time in seconds that the HR is within the normal resting range. Now, I will format the result in the specified JSON format. </thought>
<solution>
{"time_in_range": 240.0}
</solution>"""
    return prompt


def create_codeact(
    model: BaseChatModel,
    jupyter_kernel: JupyterKernel,
    *,
    prompt: Optional[str] = None,
    state_schema: StateSchemaType = CodeActState,
    verbose: bool = False,
    only_thoughts_limit: int = 5,
    use_multimodal: bool = False,
    timeout: int = 120,
) -> StateGraph:
    """Create a CodeAct agent.

    Args:
        model: The language model to use for generating code
        tools: List of tools available to the agent. Can be passed as python functions or StructuredTool instances.
        eval_fn: Function or coroutine that executes code in a sandbox. Takes code string and locals dict,
            returns a tuple of (stdout output, new variables dict)
        prompt: Optional custom system prompt. If None, uses default prompt.
            To customize default prompt you can use `create_default_prompt` helper:
            `create_default_prompt(tools, "You are a helpful assistant.")`
        state_schema: The state schema to use for the agent.
        verbose: Whether to print verbose output.
        only_thoughts_limit: Number of consecutive responses with only thoughts
        use_multimodal: Whether to use multimodal capabilities for image processing.
        timeout: Timeout in seconds for code execution.
    Returns:
        A StateGraph implementing the CodeAct architecture
    """

    prompt = create_default_prompt(prompt)

    def call_model(state: StateSchema) -> Command:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = model.invoke(messages)
        response = convert_reasoning_to_text(response)
        content = get_content_from_response(response)
        # Extract and combine all code blocks
        code = extract_combine_xml_blocks(content, tag="execute")
        ans = extract_combine_xml_blocks(content, tag="solution")
        if code:
            return Command(
                goto="sandbox",
                update={"messages": [response], "script": code, "count_thoughts": 0},
            )
        elif ans:
            return Command(
                update={"messages": [response], "script": None, "final_answer": ans},
                goto=END,
            )
        else:
            # no code block, end the loop and respond to the user
            if state.get("count_thoughts", 0) + 1 >= only_thoughts_limit:
                logger.error(
                    "Reached maximum number of consecutive thoughts without code. Ending the agent execution."
                )
                return Command(
                    update={"messages": [response], "script": None}, goto=END
                )
            if not content:  # could be empty response
                logger.error(
                    "Received empty response from the model. Ending the agent execution without update messages."
                )
                return Command(goto=END)
            return Command(
                update={
                    "messages": [response],
                    "script": None,
                    "count_thoughts": state.get("count_thoughts", 0) + 1,
                },
                goto="call_model",
            )

    async def sandbox(state: StateSchema):
        # Execute the script in the sandbox
        if verbose:
            console.print(
                Panel(
                    Syntax(state["script"], "python", word_wrap=True),
                    title="Script",
                    title_align="left",
                )
            )
        output = await jupyter_kernel.execute(state["script"], timeout=timeout)
        imgs, output = extract_md_img(output)  # remove images
        if not use_multimodal:
            return {
                "messages": [{"role": "user", "content": output}],
            }
        else:
            img_urls = [md_img_to_url(img) for img in imgs]
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": output},
                            *[
                                {"type": "image_url", "image_url": {"url": image_url}}
                                for image_url in img_urls
                                if image_url is not None
                            ],
                        ],
                    }
                ]
            }

    agent = StateGraph(state_schema)
    agent.add_node(call_model, destinations=(END, "sandbox", "call_model"))
    agent.add_node(sandbox)
    agent.add_edge(START, "call_model")
    agent.add_edge("sandbox", "call_model")
    return agent


async def query_codeact(
    query: str,
    model_name: str,
    verbose: bool = False,
    query_id: Optional[str] = None,
    only_thoughts_limit: int = 30,
    use_multimodal: bool = False,
    timeout: int = 120,
):
    model = init_chat_model(f"{model_name}")
    work_dir = Path("./agent_working")
    if query_id is not None:
        work_dir = work_dir / query_id
    work_dir.mkdir(exist_ok=True, parents=True)
    jk = JupyterLocalKernel("test", work_dir=work_dir)
    builder = create_codeact(
        model,
        jk,
        verbose=verbose,
        only_thoughts_limit=only_thoughts_limit,
        use_multimodal=use_multimodal,
        timeout=timeout,
    )
    agent = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"{uuid4()}"}, "recursion_limit": 50}
    out = {'messages': []}
    try:
        out = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}, config=config
        )
    except Exception as e:
        logger.error(f"error: {e}")
        try:
            out = agent.get_state(config).values
        except Exception as e:
            logger.error(f"error in getting state: {e}")
        out["messages"].append(AIMessage(content=f"Error: {e}"))  # return error message
    await jk.shutdown_async()
    return out
