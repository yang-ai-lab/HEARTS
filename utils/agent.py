"""
adapted from https://github.com/xingyaoww/code-act/blob/main/scripts/chat/code_execution/jupyter.py
"""

import asyncio
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Empty
from typing import Optional
from uuid import uuid4

import jupyter_client
import tornado
from langchain_core.messages import AIMessage
from loguru import logger
from tornado.escape import json_decode, json_encode, url_escape
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.ioloop import PeriodicCallback
from tornado.websocket import WebSocketClosedError, websocket_connect


def get_content_from_response(response: AIMessage) -> str:
    """Extract text content from an AIMessage response.

    Handles both string and list content types, prioritizing text content
    and falling back to reasoning content if no text is available.

    Args:
        response: The AIMessage response from the language model

    Returns:
        The extracted text content as a string

    Raises:
        ValueError: If the response content type is unsupported
    """
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        texts = []
        for item in response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        if texts:  # NOTE: try to get from "text" part only
            return "\n".join(texts)
        else:  # NOTE: if there's no "text" part, use reasoning content
            for item in response.content:
                # Question: should we combine reasoning content here? For now, just combine because aws kimi might only generate reasoning
                if isinstance(item, dict) and item.get("type") == "reasoning_content":
                    texts.append(item.get("reasoning_content", {}).get("text", ""))
            return "\n".join(texts)
    else:
        raise ValueError("Unsupported response content type.")


def convert_reasoning_to_text(response: AIMessage) -> AIMessage:
    """
    Convert reasoning-only content in an AIMessage to text content.

    If the response content is a list consisting entirely of reasoning_content items,
    this function converts each item to a text type for compatibility.

    Args:
        response: The AIMessage response from the language model.

    Returns:
        The modified AIMessage with reasoning content converted to text if applicable.
    """
    if isinstance(response.content, list):
        if len(response.content) > 0:
            if all(
                isinstance(item, dict) and item.get("type") == "reasoning_content"
                for item in response.content
            ):
                # Convert reasoning_content to text type
                new_content = []
                for item in response.content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "reasoning_content"
                    ):
                        new_content.append(
                            {
                                "type": "text",
                                "text": item.get("reasoning_content", {}).get(
                                    "text", ""
                                ),
                            }
                        )
                response.content = new_content
    return response


def strip_ansi(o: str) -> str:
    """Remove ANSI escape sequences from a string."""
    pattern = re.compile(r"\x1b\[[0-9;]*m")
    stripped = pattern.sub("", o)
    return stripped


class JupyterKernel(ABC):
    """Abstract base class for Jupyter kernels defining the common API."""

    def __init__(self, convid, lang="python", work_dir=None):
        """
        Initialize a kernel.

        Args:
            convid: Conversation ID for this kernel session
            lang: Programming language for the kernel
        """
        self.convid = convid
        self.lang = lang
        self.work_dir = work_dir
        if isinstance(self.work_dir, Path):
            self.work_dir = self.work_dir.as_posix()
        self._tools_loaded = False
        self._current_dir = "./"

    def set_work_dir(self, work_dir):
        self.work_dir = work_dir
        if isinstance(self.work_dir, Path):
            self.work_dir = self.work_dir.as_posix()
        logger.debug(f"kernel {self.convid} work directory set: {self.work_dir}")

    def reset_current_dir(self):
        self._current_dir = "./"

    async def _create_work_dir(self):
        """
        Create a work directory for the kernel and change base dir to work_dir.
        """
        # set work dir in jupyter environment
        init_code = (
            f"import os\n"
            f"if not os.path.exists('{self.work_dir}'):\n"
            f"    os.mkdir('{self.work_dir}')\n"
            f"os.chdir('{self.work_dir}')\n"
            f"del os"
        )
        await self.execute(init_code)
        logger.debug(f"kernel {self.convid} work directory created: {self.work_dir}")

    @abstractmethod
    async def load_tools(
        self,
        tool_imports_set: Optional[set] = None,
        tool_source_codes_dict: dict[str, str] = None,
    ):
        """
        Load tools into the kernel if not already loaded.

        Args:
            tool_imports_set: Set of import statements
            tool_source_codes_dict: Dictionary of tool names to source code
        """
        pass

    @abstractmethod
    async def execute(self, code, timeout=60):
        """
        Execute code in the kernel.

        Args:
            code: The code to execute
            timeout: Execution timeout in seconds

        Returns:
            The execution output or error message
        """
        pass

    @abstractmethod
    async def shutdown_async(self):
        """
        Shut down the kernel and clean up resources.
        """
        pass


class JupyterRemoteKernel(JupyterKernel):
    def __init__(
        self, convid, port=8789, lang="python", heartbeat_interval=10000, work_dir=None
    ):
        """
        Initialize a Jupyter kernel client.

        Args:
            convid: Conversation ID for this kernel session
            port: Port number for the Jupyter server
            lang: Programming language for the kernel
            heartbeat_interval: Heartbeat interval in milliseconds (default: 10 seconds)
        """
        super().__init__(convid, lang, work_dir)
        url_suffix = f"localhost:{port}"
        self.base_url = f"http://{url_suffix}"
        self.base_ws_url = f"ws://{url_suffix}"
        self.kernel_id = None
        self.ws = None
        logger.info(
            f"jupyter kernel running at {url_suffix}, conversation id: {self.convid}"
        )

        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_callback = None
        self._tools_loaded = False  # Track if tools have been loaded

    async def load_tools(
        self,
        tool_imports_set: Optional[set] = None,
        tool_source_codes_dict: dict[str, str] = None,
    ):
        """
        Load tools into the kernel if not already loaded.

        This method loads imports and tool source code into the kernel.
        It uses caching to avoid reloading tools if they're already loaded.

        Args:
            tool_imports_set: Set of import statements
            tool_source_codes_dict: Dictionary of tool names to source code
        """
        # Skip if tools are already loaded
        if self._tools_loaded:
            logger.info("Tools already loaded, skipping...")
            return

        # await self.execute(r"%colors nocolor")
        # NOTE: pre-defined tools
        all_tool_code = ""
        if tool_imports_set:
            all_tool_code += "\n".join(tool_imports_set) + "\n"
        if tool_source_codes_dict:
            all_tool_code += "\n".join(tool_source_codes_dict.values()) + "\n"

        if all_tool_code.strip():  # Only execute if there's code to load
            await self.execute(all_tool_code)  # load all tools
            self._tools_loaded = True
            logger.info("Tools loaded successfully")
        else:
            logger.info("No tools to load")

    async def _send_heartbeat(self):
        if not self.ws:
            return
        try:
            self.ws.ping()
            # logger.info("Heartbeat sent...")
        except tornado.iostream.StreamClosedError:
            # logger.info("Heartbeat failed, reconnecting...")
            try:
                await self._connect()
            except ConnectionRefusedError:
                logger.error(
                    "ConnectionRefusedError: Failed to reconnect to kernel websocket - Is the kernel still running?"
                )

    async def _connect(self):
        """
        Establish connection to the Jupyter kernel with exponential backoff.

        This method creates a new kernel if one doesn't exist, and establishes
        a websocket connection to it. It uses exponential backoff for retrying
        kernel creation to be more resource-friendly.
        """
        if self.ws:
            self.ws.close()
            self.ws = None

        client = AsyncHTTPClient()
        if not self.kernel_id:
            n_tries = 5
            backoff_delay = 1  # Start with 1 second delay
            while n_tries > 0:
                try:
                    response = await client.fetch(
                        "{}/api/kernels".format(self.base_url),
                        method="POST",
                        body=json_encode({"name": self.lang}),
                    )
                    kernel = json_decode(response.body)
                    self.kernel_id = kernel["id"]
                    logger.info(f"kernel created, id: {self.kernel_id}")
                    break
                except Exception as e:
                    # kernels are not ready yet
                    n_tries -= 1
                    logger.error(f"Fail to create kernel: {e}")
                    if n_tries > 0:  # Don't sleep on the last iteration
                        await asyncio.sleep(backoff_delay)
                        backoff_delay *= 2  # Exponential backoff

            if n_tries == 0:
                raise ConnectionRefusedError("Failed to connect to kernel")

        ws_req = HTTPRequest(
            url="{}/api/kernels/{}/channels".format(
                self.base_ws_url, url_escape(self.kernel_id)
            )
        )
        self.ws = await websocket_connect(ws_req)
        logger.info("Connected to kernel websocket")

        # Setup heartbeat
        if self.heartbeat_callback:
            self.heartbeat_callback.stop()
        self.heartbeat_callback = PeriodicCallback(
            self._send_heartbeat, self.heartbeat_interval
        )
        self.heartbeat_callback.start()

    async def execute(self, code, timeout=60):
        """
        Execute code in the Jupyter kernel.

        This method sends code to the kernel for execution and waits for the results.
        It handles various message types from the kernel and processes them appropriately.

        Args:
            code: The code to execute
            timeout: Execution timeout in seconds

        Returns:
            The execution output or error message
        """
        if not self.ws:
            await self._connect()

        if self.work_dir and self._current_dir != self.work_dir:
            self._current_dir = self.work_dir
            await self._create_work_dir()

        msg_id = uuid4().hex

        # make sure ws is connected
        if self.ws.protocol is None:  # not connected
            await self._connect()  # try reconnect

        # Send message with retry logic for connection issues
        max_retries = 3
        base_delay = 1  # 1 second
        for attempt in range(max_retries + 1):
            try:
                self.ws.write_message(
                    json_encode(
                        {
                            "header": {
                                "username": "",
                                "version": "5.0",
                                "session": "",
                                "msg_id": msg_id,
                                "msg_type": "execute_request",
                            },
                            "parent_header": {},
                            "channel": "shell",
                            "content": {
                                "code": code,
                                "silent": False,
                                "store_history": False,
                                "user_expressions": {},
                                "allow_stdin": False,
                            },
                            "metadata": {},
                            "buffers": {},
                        }
                    )
                )
                break  # Success, exit the retry loop
            except (WebSocketClosedError, tornado.iostream.StreamClosedError) as e:
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2**attempt)  # 1, 2, 4 seconds
                    logger.warning(
                        f"WebSocket write failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Failed to send message after {max_retries + 1} attempts: {e}"
                    )
                    raise

        outputs = []
        start_time = asyncio.get_event_loop().time()

        async def wait_for_messages():
            execution_done = False
            while not execution_done:
                # Check for timeout before waiting for message
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise asyncio.TimeoutError()

                # Wait for message with remaining timeout
                remaining_timeout = timeout - elapsed
                try:
                    msg = await asyncio.wait_for(
                        self.ws.read_message(), remaining_timeout
                    )
                except asyncio.TimeoutError:
                    raise  # Re-raise timeout error

                if not msg:
                    continue
                msg = json_decode(msg)
                msg_type = msg["msg_type"]
                parent_msg_id = msg["parent_header"].get("msg_id", None)

                if parent_msg_id != msg_id:
                    continue

                if msg_type == "error":
                    traceback = "\n".join(msg["content"]["traceback"])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == "stream":
                    outputs.append(msg["content"]["text"])
                    logger.debug(
                        "intermediate jupyter run output\n"
                        + msg["content"]["text"].strip()
                    )
                elif msg_type in ["execute_result", "display_data"]:
                    outputs.append(msg["content"]["data"]["text/plain"])
                    if "image/png" in msg["content"]["data"]:
                        # use markdown to display image (in case of large image)
                        outputs.append(
                            f"![image](data:image/png;base64,{msg['content']['data']['image/png']})"
                        )

                elif msg_type == "execute_reply":
                    execution_done = True
            return execution_done

        async def interrupt_kernel():
            try:
                client = AsyncHTTPClient()
                interrupt_response = await client.fetch(
                    f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt",
                    method="POST",
                    body=json_encode({"kernel_id": self.kernel_id}),
                    request_timeout=10,  # Add timeout for interrupt request
                )
                logger.info(f"Kernel interrupted: {interrupt_response}")
            except Exception as e:
                logger.error(f"Failed to interrupt kernel: {e}")

        ret = ""
        try:
            execution_done = await wait_for_messages()
        except asyncio.TimeoutError:
            await interrupt_kernel()
            ret = f"[Execution timed out ({timeout} seconds).]"
            execution_done = False

        if not outputs and execution_done:
            ret = "[Code executed successfully with no output]"
        else:
            ret = "".join(outputs) + ret

        # Remove ANSI
        ret = strip_ansi(ret)

        return ret

    async def shutdown_async(self):
        """
        Shut down the Jupyter kernel and clean up resources.

        This method terminates the kernel process and closes the websocket connection.
        It also resets the tools_loaded flag to allow reloading tools in a new session.
        """
        if self.kernel_id:
            kid = self.kernel_id
            try:
                client = AsyncHTTPClient()
                await client.fetch(
                    "{}/api/kernels/{}".format(self.base_url, self.kernel_id),
                    method="DELETE",
                    request_timeout=10,  # Add timeout for shutdown request
                )
            except Exception as e:
                logger.error(f"Error shutting down kernel {kid}: {e}")
            finally:
                self.kernel_id = None
                self._tools_loaded = False  # Reset tools loaded flag
                if self.ws:
                    self.ws.close()
                    self.ws = None
                logger.info(f"kernel with id: {kid} is closed")


class JupyterLocalKernel(JupyterKernel):
    def __init__(
        self, convid, port=None, lang="python", heartbeat_interval=None, work_dir=None
    ):
        """
        local jupyter kernel

        Args:
            convid: Conversation ID for this kernel session
            port: Port number (not used in local implementation)
            lang: Programming language for the kernel
            heartbeat_interval: Heartbeat interval (not used in local implementation)
        """
        super().__init__(convid, lang, work_dir)
        self.kernel_manager = None
        self.kernel_client = None

        # logger.info(f"jupyter kernel running at local, conversation id: {self.convid}")

    async def load_tools(
        self,
        tool_imports_set: Optional[set] = None,
        tool_source_codes_dict: dict[str, str] = None,
    ):
        """
        Load tools into the kernel if not already loaded.

        This method loads imports and tool source code into the kernel.
        It uses caching to avoid reloading tools if they're already loaded.

        Args:
            tool_imports_set: Set of import statements
            tool_source_codes_dict: Dictionary of tool names to source code
        """
        # Skip if tools are already loaded
        if self._tools_loaded:
            logger.info("Tools already loaded, skipping...")
            return

        # NOTE: pre-defined tools
        all_tool_code = ""
        if tool_imports_set:
            all_tool_code += "\n".join(tool_imports_set) + "\n"
        if tool_source_codes_dict:
            all_tool_code += "\n".join(tool_source_codes_dict.values()) + "\n"

        if all_tool_code.strip():  # Only execute if there's code to load
            await self.execute(all_tool_code)  # load all tools
            self._tools_loaded = True
            logger.info("Tools loaded successfully")
        else:
            logger.info("No tools to load")

    async def _connect(self):
        """
        Establish connection to the Jupyter kernel.

        This method creates a new kernel manager and starts a kernel client.
        """
        if self.kernel_client is None:
            (
                self.kernel_manager,
                self.kernel_client,
            ) = await jupyter_client.manager.start_new_async_kernel(
                kernel_name="python3"
            )
            logger.debug(f"kernel with id: {self.kernel_manager.kernel_id} started")

        # TODO: re-connect logic?

    async def execute(self, code, timeout=60):
        """
        Execute code in the Jupyter kernel.

        This method sends code to the kernel for execution and waits for the results.
        It handles various message types from the kernel and processes them appropriately.

        Args:
            code: The code to execute
            timeout: Execution timeout in seconds

        Returns:
            The execution output or error message
        """
        if not self.kernel_client:
            await self._connect()

        if self.work_dir and self._current_dir != self.work_dir:
            self._current_dir = self.work_dir
            await self._create_work_dir()

        if self.kernel_client is None or not await self.kernel_client.is_alive():
            logger.warning(
                f"Kernel {self.kernel_manager.kernel_id} is not alive, restarting it."
            )
            await self.shutdown_async()
            await self._connect()
            self._current_dir = self.work_dir
            await self._create_work_dir()

        outputs = []
        start_time = asyncio.get_event_loop().time()
        msg_id = self.kernel_client.execute(code)

        async def wait_for_messages():
            execution_done = False
            while not execution_done:
                # Check for timeout before waiting for message
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise asyncio.TimeoutError()

                # Wait for message with remaining timeout
                remaining_timeout = timeout - elapsed
                try:
                    msg = await self.kernel_client.get_iopub_msg(
                        timeout=remaining_timeout
                    )
                except:
                    raise  # Re-raise timeout error

                if not msg:
                    continue
                msg_type = msg["msg_type"]
                parent_msg_id = msg["parent_header"].get("msg_id", None)

                if parent_msg_id != msg_id:
                    continue

                if msg_type == "error":
                    traceback = "\n".join(msg["content"]["traceback"])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == "stream":
                    outputs.append(msg["content"]["text"])
                elif msg_type in ["execute_result", "display_data"]:
                    text = msg["content"]["data"]["text/plain"]
                    outputs.append(text)
                    if "image/png" in msg["content"]["data"]:
                        # use markdown to display image (in case of large image)
                        outputs.append(
                            f"![image](data:image/png;base64,{msg['content']['data']['image/png']})"
                        )

                elif (
                    msg_type == "status"
                    and msg["content"].get("execution_state") == "idle"
                ):
                    execution_done = True
            return execution_done

        async def interrupt_kernel():
            try:
                await self.kernel_manager.interrupt_kernel()
                logger.warning("Kernel interrupted")
            except Exception as e:
                logger.error(f"Failed to interrupt kernel: {e}")

        ret = ""
        try:
            execution_done = await wait_for_messages()
        except Empty:
            await interrupt_kernel()
            ret = f"[Execution timed out ({timeout} seconds).]"
            execution_done = False

        if not outputs and execution_done:
            ret = "[Code executed successfully with no output]"
        else:
            ret = "".join(outputs) + ret

        # Remove ANSI
        ret = strip_ansi(ret)

        return ret

    async def shutdown_async(self):
        """
        Shut down the Jupyter kernel and clean up resources.

        This method terminates the kernel process and closes the client connection.
        It also resets the tools_loaded flag to allow reloading tools in a new session.
        """
        try:
            self.reset_current_dir()
            if self.kernel_client:
                self.kernel_client.shutdown()
                del self.kernel_client
                self.kernel_client = None

            if self.kernel_manager:
                kernel_id = self.kernel_manager.kernel_id
                await self.kernel_manager.shutdown_kernel(now=True)
                del self.kernel_manager
                self.kernel_manager = None
                self._tools_loaded = False  # Reset tools loaded flag
                logger.debug(f"kernel with id: {kernel_id} is closed")
        except Exception as e:
            logger.error(f"Error shutting down kernel: {e}")


async def _test():
    convid = str(uuid.uuid4())
    # jk = JupyterRemoteKernel(convid)
    jk = JupyterLocalKernel(convid)
    code = """
import psutil
import platform

# Get CPU information
cpu_count = psutil.cpu_count(logical=False)
cpu_count_logical = psutil.cpu_count(logical=True)
cpu_freq = psutil.cpu_freq()

# Get memory information
memory = psutil.virtual_memory()

# Get disk information
disk = psutil.disk_usage('.')

# Get system information
system = platform.system()
release = platform.release()
version = platform.version()
machine = platform.machine()
processor = platform.processor()

import os
import multiprocessing

def get_usable_cpu_count():
    # SLURM
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    elif "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        return int(os.environ["SLURM_JOB_CPUS_PER_NODE"])

    # SGE / Grid Engine
    elif "NSLOTS" in os.environ:
        return int(os.environ["NSLOTS"])

    # LSF
    elif "LSB_DJOB_NUMPROC" in os.environ:
        return int(os.environ["LSB_DJOB_NUMPROC"])

    # PBS / Torque
    elif "PBS_NP" in os.environ:
        return int(os.environ["PBS_NP"])

    # Default to available logical CPUs
    try:
        import psutil
        return len(psutil.Process().cpu_affinity())
    except (ImportError, AttributeError):
        return multiprocessing.cpu_count()

print("usable CPU num:",get_usable_cpu_count())
print(f"System: {system} {release} {version}")
print(f"Machine: {machine}")
print(f"Processor: {processor}")
print(f"CPU Cores (Physical): {cpu_count}")
print(f"CPU Cores (Logical): {cpu_count_logical}")
print(f"CPU Frequency: {cpu_freq.max} MHz")
print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
print(f"Memory Usage: {memory.percent}%")
print(f"Total Disk Space: {disk.total / (1024**3):.2f} GB")
print(f"Used Disk Space: {disk.used / (1024**3):.2f} GB")
print(f"Free Disk Space: {disk.free / (1024**3):.2f} GB")
    """
    print(await jk.execute(code))

    print(await jk.execute("!lscpu"))

    await jk.shutdown_async()

    return


__all__ = [
    "JupyterRemoteKernel",
    "JupyterLocalKernel",
    "get_content_from_response",
    "strip_ansi",
    "JupyterKernel",
]
