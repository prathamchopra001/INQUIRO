import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from pathlib import Path


class NotebookManager:
    """Creates Jupyter notebooks for traceability."""

    def __init__(self, output_dir: str = "./outputs/notebooks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_notebook(self, title: str, task_description: str, cycle: int) -> nbformat.NotebookNode:
        nb = new_notebook()
        header = (
            f"# {title}\n\n"
            f"**Cycle:** {cycle}\n\n"
            f"{task_description}"
        )
        nb.cells.append(new_markdown_cell(header))
        return nb

    def add_code_cell(self, notebook: nbformat.NotebookNode, code: str, output: str = None) -> int:
        cell = new_code_cell(source=code)
        if output is not None:
            out = new_output(output_type="stream", name="stdout", text=output)
            cell.outputs = [out]

        notebook.cells.append(cell)
        return len(notebook.cells) - 1

    def add_markdown_cell(self, notebook: nbformat.NotebookNode, content: str) -> int:
        cell = new_markdown_cell(source=content)
        notebook.cells.append(cell)
        return len(notebook.cells) - 1

    def save_notebook(self, notebook: nbformat.NotebookNode, filename: str = None) -> str:
        if filename is None:
            existing = list(self.output_dir.glob("analysis_*.ipynb"))
            next_num = len(existing) + 1
            filename = f"analysis_{next_num:03d}.ipynb"

        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        return str(path)

