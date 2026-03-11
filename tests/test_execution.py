"""
Tests for execution modules (DockerExecutor and NotebookManager).
"""

import pytest
import os
import nbformat
import subprocess
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from src.execution.docker_executor import DockerExecutor, ExecutionResult
from src.execution.notebook_manager import NotebookManager


# =============================================================================
# NOTEBOOK MANAGER TESTS
# =============================================================================

class TestNotebookManager:
    """Tests for NotebookManager."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "notebooks"

    @pytest.fixture
    def manager(self, output_dir):
        return NotebookManager(output_dir=str(output_dir))

    def test_create_notebook(self, manager):
        """Test creating a new notebook."""
        nb = manager.create_notebook("Test Title", "Do task X", 1)
        assert len(nb.cells) == 1
        assert nb.cells[0].cell_type == "markdown"
        assert "# Test Title" in nb.cells[0].source
        assert "**Cycle:** 1" in nb.cells[0].source
        assert "Do task X" in nb.cells[0].source

    def test_add_code_cell(self, manager):
        """Test adding a code cell."""
        nb = manager.create_notebook("Test", "Task", 1)
        idx = manager.add_code_cell(nb, "print('hello')", output="hello")

        assert idx == 1  # 0 is header, 1 is code
        assert len(nb.cells) == 2
        assert nb.cells[1].cell_type == "code"
        assert nb.cells[1].source == "print('hello')"
        assert nb.cells[1].outputs[0].text == "hello"

    def test_add_markdown_cell(self, manager):
        """Test adding a markdown cell."""
        nb = manager.create_notebook("Test", "Task", 1)
        idx = manager.add_markdown_cell(nb, "## Analysis")

        assert idx == 1
        assert len(nb.cells) == 2
        assert nb.cells[1].cell_type == "markdown"
        assert nb.cells[1].source == "## Analysis"

    def test_save_notebook_auto_name(self, manager):
        """Test saving notebook with auto-generated name."""
        nb = manager.create_notebook("Test", "Task", 1)
        path = manager.save_notebook(nb)

        assert "analysis_001.ipynb" in path
        assert Path(path).exists()

        # Second one should be 002
        path2 = manager.save_notebook(nb)
        assert "analysis_002.ipynb" in path2

    def test_save_notebook_custom_name(self, manager):
        """Test saving notebook with custom name."""
        nb = manager.create_notebook("Test", "Task", 1)
        path = manager.save_notebook(nb, filename="my_analysis.ipynb")

        assert "my_analysis.ipynb" in path
        assert Path(path).exists()

        # Verify content
        with open(path, 'r', encoding='utf-8') as f:
            read_nb = nbformat.read(f, as_version=4)
            assert len(read_nb.cells) == 1


# =============================================================================
# DOCKER EXECUTOR TESTS
# =============================================================================

class TestDockerExecutor:
    """Tests for DockerExecutor."""

    @patch("subprocess.run")
    def test_execute_code_success(self, mock_run, tmp_path):
        """Test successful execution."""
        executor = DockerExecutor()

        # Mock successful result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = executor.execute_code("print('hello')", output_path=str(tmp_path))

        assert result.success is True
        assert result.stdout == "hello"
        assert result.exit_code == 0
        assert result.timed_out is False

        # Verify docker command structure
        args, _ = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert executor.image in cmd

    @patch("subprocess.run")
    def test_execute_code_timeout(self, mock_run, tmp_path):
        """Test execution timeout."""
        executor = DockerExecutor()

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=300)

        result = executor.execute_code("while True: pass", output_path=str(tmp_path))

        assert result.success is False
        assert result.timed_out is True
        assert "timed out" in result.stderr

    @patch("subprocess.run")
    def test_execute_code_error(self, mock_run, tmp_path):
        """Test execution error (non-zero exit code)."""
        executor = DockerExecutor()

        # Mock failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "SyntaxError"
        mock_run.return_value = mock_result

        result = executor.execute_code("invalid code", output_path=str(tmp_path))

        assert result.success is False
        assert result.exit_code == 1
        assert "SyntaxError" in result.stderr

    @patch("subprocess.run")
    def test_execute_code_exception(self, mock_run, tmp_path):
        """Test unexpected exception during execution."""
        executor = DockerExecutor()

        # Mock unexpected error
        mock_run.side_effect = Exception("Docker daemon not running")

        result = executor.execute_code("print('hello')", output_path=str(tmp_path))

        assert result.success is False
        assert "Docker daemon not running" in result.stderr
        assert result.timed_out is False
