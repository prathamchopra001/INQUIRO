"""
Native Python Executor - Fallback when Docker is unavailable.

This executor runs code in a subprocess on the local machine.
It's less secure than Docker (no sandboxing) but allows INQUIRO
to function when Docker isn't available.

WARNING: This executes arbitrary Python code on your system.
Only use with trusted code generation (like from your own LLM).

Usage:
    from src.execution.native_executor import NativeExecutor
    
    executor = NativeExecutor()
    result = executor.execute_code(code, data_path="./data.csv")
"""

import subprocess
import tempfile
import time
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class ExecutionResult:
    """Result of code execution - same interface as DockerExecutor."""
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    timed_out: bool
    figures: list[str]
    execution_time: float


class NativeExecutor:
    """
    Executes Python code in a local subprocess.
    
    This is a fallback for when Docker is unavailable.
    Same interface as DockerExecutor for drop-in replacement.
    """
    
    def __init__(self, timeout: int = 300, python_path: str = None):
        """
        Args:
            timeout: Maximum execution time in seconds
            python_path: Path to Python interpreter (default: sys.executable)
        """
        self.timeout = timeout
        self.python_path = python_path or sys.executable
    
    def execute_code(
        self, 
        code: str, 
        data_path: str = None, 
        output_path: str = None
    ) -> ExecutionResult:
        """
        Execute Python code in a subprocess.
        
        Args:
            code: Python code to execute
            data_path: Path to data file/directory (will be available as DATA_PATH env var)
            output_path: Directory for output files (created if needed)
            
        Returns:
            ExecutionResult with stdout, stderr, figures, etc.
        """
        start_time = time.time()
        
        # Create output directory
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="inquiro_native_out_")
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare code with path substitutions
        # Replace Docker-style paths with local paths
        modified_code = self._adapt_code_for_native(code, data_path, output_path)
        
        # Write code to temp file
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", 
            delete=False, 
            mode="w", 
            encoding="utf-8"
        )
        try:
            tmp.write(modified_code)
            tmp.flush()
            script_path = tmp.name
        finally:
            tmp.close()
        
        try:
            # Set up environment
            env = os.environ.copy()
            if data_path:
                env["DATA_PATH"] = os.path.abspath(data_path)
            env["OUTPUT_PATH"] = os.path.abspath(output_path)
            
            # Execute
            result = subprocess.run(
                [self.python_path, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=os.path.dirname(script_path),
            )
            
            # Collect output files (figures, CSVs, etc.)
            output_files = []
            out_dir = Path(output_path)
            if out_dir.exists():
                for f in out_dir.iterdir():
                    if f.is_file():
                        output_files.append(str(f))
            
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                success=(result.returncode == 0),
                timed_out=False,
                figures=output_files,
                execution_time=time.time() - start_time,
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout} seconds.",
                exit_code=-1,
                success=False,
                timed_out=True,
                figures=[],
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                success=False,
                timed_out=False,
                figures=[],
                execution_time=time.time() - start_time,
            )
            
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass
    
    def _adapt_code_for_native(
        self, 
        code: str, 
        data_path: str, 
        output_path: str
    ) -> str:
        """
        Adapt Docker-style code for native execution.
        
        Replaces:
        - /app/data/ → actual data path
        - /app/outputs/ → actual output path
        """
        modified = code
        
        # Replace Docker mount paths with actual paths
        if data_path:
            abs_data = os.path.abspath(data_path)
            if os.path.isfile(abs_data):
                # If data_path is a file, use its directory for /app/data/
                data_dir = os.path.dirname(abs_data)
                modified = modified.replace("/app/data/", data_dir.replace("\\", "/") + "/")
                # Also handle the specific file reference
                filename = os.path.basename(abs_data)
                modified = modified.replace(
                    f"/app/data/{filename}", 
                    abs_data.replace("\\", "/")
                )
            else:
                # data_path is a directory
                modified = modified.replace("/app/data/", abs_data.replace("\\", "/") + "/")
        
        if output_path:
            abs_output = os.path.abspath(output_path)
            modified = modified.replace("/app/outputs/", abs_output.replace("\\", "/") + "/")
            modified = modified.replace("/app/outputs", abs_output.replace("\\", "/"))
        
        return modified
    
    def execute_code_with_resolver(
        self,
        code: str,
        data_path: str = None,
        output_path: str = None,
        max_package_retries: int = 2,
    ) -> ExecutionResult:
        """
        Execute code with automatic package installation on ImportError.
        
        Same interface as DockerExecutor.execute_code_with_resolver().
        """
        from src.execution.package_resolver import PackageResolver
        
        resolver = PackageResolver()
        current_code = code
        
        for attempt in range(1, max_package_retries + 2):
            result = self.execute_code(current_code, data_path, output_path)
            
            if result.success:
                return result
            
            # Check if failure was due to missing package
            is_import_error = (
                "ModuleNotFoundError" in result.stderr or
                "No module named" in result.stderr or
                "cannot import name" in result.stderr
            )
            
            if is_import_error and attempt <= max_package_retries:
                patched = resolver.patch_code(current_code, result.stderr)
                if patched != current_code:
                    import logging
                    logging.getLogger(__name__).info(
                        f"PackageResolver: retrying with auto-install "
                        f"(attempt {attempt}/{max_package_retries})"
                    )
                    current_code = patched
                    continue
            
            return result
        
        return result
    
    # Dummy methods to match DockerExecutor interface
    def warm_pool(self) -> None:
        """No-op for native executor (no container pool)."""
        pass
    
    def drain_pool(self) -> None:
        """No-op for native executor (no container pool)."""
        pass


def is_docker_available() -> bool:
    """
    Check if Docker is available and running.
    
    Returns:
        True if Docker can execute commands, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def get_executor(prefer_docker: bool = True, **kwargs):
    """
    Get the appropriate executor based on availability.
    
    Args:
        prefer_docker: If True, use Docker when available
        **kwargs: Passed to executor constructor
        
    Returns:
        DockerExecutor if Docker is available and preferred, else NativeExecutor
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if prefer_docker and is_docker_available():
        from src.execution.docker_executor import DockerExecutor
        logger.info("Using Docker executor (sandboxed)")
        return DockerExecutor(**kwargs)
    else:
        if prefer_docker:
            logger.warning(
                "Docker not available - using native Python executor. "
                "Code will run without sandboxing."
            )
        else:
            logger.info("Using native Python executor (Docker disabled)")
        return NativeExecutor(**kwargs)
