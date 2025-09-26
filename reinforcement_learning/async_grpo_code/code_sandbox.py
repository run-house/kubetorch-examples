import asyncio
import base64
import json
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeSandbox:
    """Code execution environment using SWE-ReX for sandboxed execution"""

    def __init__(self):
        from swerex.runtime.local import LocalRuntime

        self.runtime = None
        self._current_tasks = None
        try:
            self.runtime = LocalRuntime()

            logger.info("Code sandbox initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup sandbox: {e}")
            raise

    def execute_code(self, code: str, timeout: int = 30) -> Dict:
        from swerex.runtime.local import Command

        """Execute Python code in the sandbox environment."""
        encoded_code = base64.b64encode(code.encode()).decode()
        try:
            wrapper_script = f"""
import sys, io, json, traceback, base64
from contextlib import redirect_stdout, redirect_stderr

code = base64.b64decode("{encoded_code}").decode()

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()

try:
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exec(code, {{'__name__': '__main__'}})
    result = {{
        "success": True,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue()
    }}
except Exception as e:
    result = {{
        "success": False,
        "stdout": stdout_buf.getvalue(),
        "stderr": traceback.format_exc()
    }}

print(json.dumps(result))
"""

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            response = loop.run_until_complete(
                self.runtime.execute(
                    Command(
                        command=["python3", "-c", wrapper_script],
                        shell=False,
                        timeout=timeout,
                    )
                )
            )
            return json.loads(response.stdout.strip())

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Code execution failed " + str(e),
            }
        finally:
            if loop:
                # Cancel all pending tasks
                for task in asyncio.all_tasks(loop):
                    if not task.done():
                        task.cancel()
                # Clean up the loop
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
                except:
                    pass


if __name__ == "__main__":
    import kubetorch as kt

    cpus = kt.Compute(
        cpus="0.5",
        image=kt.Image().run_bash("uv pip install  --system pandas numpy swe-rex"),
    ).autoscale(min_scale=1, max_scale=5, concurrency=1, metric="concurrency")

    agent = kt.cls(CodeSandbox).to(cpus)
    result = agent.execute_code('print("hello world")')
    print(result)
