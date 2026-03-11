import subprocess
import time
import tempfile
import threading
import os
from dataclasses import dataclass
from pathlib import Path

from src.execution.package_resolver import PackageResolver


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    timed_out: bool
    figures: list[str]
    execution_time: float


class DockerExecutor:
    """Executes code in isolated Docker containers."""

    def __init__(self, image: str = "inquiro-sandbox", timeout: int = 300, pool_size: int = 2):
        self.image = image
        self.timeout = timeout
        self.pool_size = pool_size
        self._resolver = PackageResolver()
        # D2: Container pool — pre-warmed container IDs ready to exec into
        self._pool: list[str] = []
        self._pool_lock = threading.Lock()
        self._pool_ready = False

    def execute_code(self, code: str, data_path: str = None, output_path: str = None) -> ExecutionResult:
        """Execute code, using pooled container if available, fresh container otherwise."""
        start_time = time.time()
        import logging
        logger = logging.getLogger(__name__)

        # Each execution gets its own output dir to prevent parallel collisions
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="inquiro_out_")
        os.makedirs(output_path, exist_ok=True)

        tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
        try:
            tmp.write(code)
            tmp.flush()
            script_path = tmp.name
        finally:
            tmp.close()

        # Try to use a pooled container first (faster execution)
        container_id = self._get_pooled_container()
        if container_id:
            try:
                result = self._execute_in_pooled_container(
                    container_id, script_path, data_path, output_path
                )
                return result
            except Exception as e:
                logger.warning(f"Pooled execution failed: {e}, falling back to fresh container")
                self._cleanup_container(container_id)
            finally:
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

        # Fallback: use fresh container (original behavior)
        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                "--memory",
                "4g",
                "--cpus",
                "2",
                "--network",
                "none",
            ]

            # mount data (parent dir) if provided
            if data_path:
                abs_path = os.path.abspath(data_path)
                parent_dir = os.path.dirname(abs_path)
                cmd += ["-v", f"{parent_dir}:/app/data:ro"]

            cmd += ["-v", f"{os.path.abspath(output_path)}:/app/outputs:rw"]
            cmd += ["-v", f"{script_path}:/app/script.py:ro"]
            cmd += [self.image, "python3", "/app/script.py"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            stdout = result.stdout
            stderr = result.stderr

            # collect outputs
            output_files: list[str] = []
            out_dir = Path(output_path)
            if out_dir.exists():
                for f in out_dir.iterdir():
                    if f.is_file():
                        output_files.append(str(f))

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                success=(result.returncode == 0),
                timed_out=False,
                figures=output_files,
                execution_time=time.time() - start_time,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="Execution timed out.",
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

    def _is_container_running(self, container_id: str) -> bool:
        """Check if a container is still running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and "true" in result.stdout.lower()
        except Exception:
            return False

    def _get_pooled_container(self) -> str | None:
        """Get an available container from the pool, verifying it's still running."""
        import logging
        logger = logging.getLogger(__name__)
        
        with self._pool_lock:
            stale_count = 0
            while self._pool_ready and self._pool:
                container_id = self._pool.pop()
                if self._is_container_running(container_id):
                    return container_id
                # Container is stale, clean it up
                stale_count += 1
                logger.debug(f"Container {container_id[:12]} is stale, cleaning up")
                self._cleanup_container(container_id)
            
            # If we found stale containers and pool is now empty, trigger replenishment
            if stale_count > 0:
                logger.info(f"Pool exhausted ({stale_count} stale containers), triggering replenishment")
                self._pool_ready = False  # Allow warm_pool to run again
                threading.Thread(target=self._warm_single_container, daemon=True).start()
        return None

    def _warm_single_container(self) -> None:
        """Create a single container for immediate use."""
        import logging
        logger = logging.getLogger(__name__)
        try:
            result = subprocess.run(
                ["docker", "create", "--network", "none",
                 "--memory", "4g", "--cpus", "2",
                 self.image, "sleep", "infinity"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                container_id = result.stdout.strip()
                subprocess.run(
                    ["docker", "start", container_id],
                    capture_output=True, timeout=15
                )
                with self._pool_lock:
                    self._pool.append(container_id)
                    self._pool_ready = True
                logger.info(f"Container pool: replenished with {container_id[:12]}")
        except Exception as e:
            logger.warning(f"Container replenishment failed: {e}")

    def _return_to_pool(self, container_id: str) -> None:
        """Return a container to the pool after use."""
        try:
            # Clean up the container for reuse
            subprocess.run(
                ["docker", "exec", container_id, "rm", "-rf", "/app/script.py"],
                capture_output=True, timeout=5
            )
            with self._pool_lock:
                self._pool.append(container_id)
        except Exception:
            self._cleanup_container(container_id)

    def _cleanup_container(self, container_id: str) -> None:
        """Remove a container that can't be reused."""
        try:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
        except Exception:
            pass

    def _execute_in_pooled_container(
        self, container_id: str, script_path: str, data_path: str, output_path: str
    ) -> ExecutionResult:
        """Execute code in a pre-warmed pooled container."""
        import logging
        logger = logging.getLogger(__name__)
        start_time = time.time()

        try:
            # Copy script into container
            subprocess.run(
                ["docker", "cp", script_path, f"{container_id}:/app/script.py"],
                check=True, timeout=10
            )

            # Copy data if provided
            if data_path and os.path.exists(data_path):
                abs_path = os.path.abspath(data_path)
                if os.path.isfile(abs_path):
                    subprocess.run(
                        ["docker", "cp", abs_path, f"{container_id}:/app/data/"],
                        capture_output=True, timeout=30
                    )
                else:
                    for item in os.listdir(abs_path):
                        item_path = os.path.join(abs_path, item)
                        subprocess.run(
                            ["docker", "cp", item_path, f"{container_id}:/app/data/"],
                            capture_output=True, timeout=30
                        )

            # Execute the script
            result = subprocess.run(
                ["docker", "exec", container_id, "python3", "/app/script.py"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Copy outputs back
            subprocess.run(
                ["docker", "cp", f"{container_id}:/app/outputs/.", output_path],
                capture_output=True, timeout=30
            )

            # Find generated figures
            output_files = []
            out_dir = Path(output_path)
            if out_dir.exists():
                for f in out_dir.iterdir():
                    if f.is_file():
                        output_files.append(str(f))

            # Return container to pool for reuse
            self._return_to_pool(container_id)
            logger.debug(f"Pooled execution completed in {time.time() - start_time:.1f}s")

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
            self._cleanup_container(container_id)
            return ExecutionResult(
                stdout="",
                stderr="Execution timed out (pooled container).",
                exit_code=-1,
                success=False,
                timed_out=True,
                figures=[],
                execution_time=time.time() - start_time,
            )

    def execute_code_with_resolver(
        self,
        code: str,
        data_path: str = None,
        output_path: str = "./outputs",
        max_package_retries: int = 2,
    ) -> ExecutionResult:
        """
        Execute code with automatic package installation on ImportError.

        Wraps execute_code with D3 package resolver:
        if execution fails with a missing module error,
        automatically prepend pip install and retry.

        Args:
            code:                Original code to execute
            data_path:           Path to data directory
            output_path:         Path for outputs
            max_package_retries: Max auto-install attempts

        Returns:
            ExecutionResult from the final attempt
        """
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
                patched = self._resolver.patch_code(current_code, result.stderr)
                if patched != current_code:
                    import logging
                    logging.getLogger(__name__).info(
                        f"PackageResolver: retrying with auto-install "
                        f"(attempt {attempt}/{max_package_retries})"
                    )
                    current_code = patched
                    continue

            # Not an import error or no packages detected — return as-is
            return result

        return result

    # =========================================================================
    # D2: Container Pool
    # =========================================================================

    def warm_pool(self) -> None:
        """
        Pre-create containers in the background so they're ready to use.
        Containers are created with --network none and kept idle.
        """
        import logging
        logger = logging.getLogger(__name__)

        def _warm():
            for i in range(self.pool_size):
                try:
                    result = subprocess.run(
                        ["docker", "create", "--network", "none",
                         "--memory", "4g", "--cpus", "2",
                         self.image, "sleep", "infinity"],  # Keep alive indefinitely
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        container_id = result.stdout.strip()
                        subprocess.run(
                            ["docker", "start", container_id],
                            capture_output=True, timeout=15
                        )
                        with self._pool_lock:
                            self._pool.append(container_id)
                        logger.debug(f"Container pool: warmed {container_id[:12]}")
                except Exception as e:
                    logger.debug(f"Container pool warm-up failed: {e}")

            with self._pool_lock:
                self._pool_ready = True
            logger.info(f"Container pool ready: {len(self._pool)} containers")

        # Run in background so it doesn't block startup
        threading.Thread(target=_warm, daemon=True).start()

    def drain_pool(self) -> None:
        """Stop and remove all pooled containers."""
        with self._pool_lock:
            for cid in self._pool:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", cid],
                        capture_output=True, timeout=10
                    )
                except Exception:
                    pass
            self._pool.clear()
            self._pool_ready = False
