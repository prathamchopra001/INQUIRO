# -*- coding: utf-8 -*-
"""Entry point for the INQUIRO CLI."""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import cli

if __name__ == "__main__":
    cli()