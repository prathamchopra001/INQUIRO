# -*- coding: utf-8 -*-
"""
INQUIRO Command Line Interface
"""
import click
import logging
import sys
from pathlib import Path
from datetime import datetime


def _setup_logging(verbose: bool):
    """Configure logging for CLI runs."""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"inquiro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


@click.group()
def cli():
    """INQUIRO - Autonomous AI Research System"""
    pass


@cli.command()
@click.option("--objective", "-o", required=True,
              help="Research objective describing what to investigate.")
@click.option("--data", "-d", required=True,
              type=click.Path(exists=True),
              help="Path to the dataset CSV file.")
@click.option("--cycles", "-c", default=5, show_default=True,
              help="Number of research cycles to run.")
@click.option("--tasks", "-t", default=3, show_default=True,
              help="Number of tasks per cycle (data + literature).")
@click.option("--workers", "-w", default=2, show_default=True,
              help="Number of parallel workers.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable verbose debug logging.")
@click.option("--seed-papers", "-s", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Path to folder of seed PDFs to pre-load into RAG.")
@click.option("--mode", "-m", default=None,
              type=click.Choice(["literature", "data", "full", "auto"], case_sensitive=False),
              help="Run mode: literature, data, full, or auto (default: auto).")


def run(objective, data, cycles, tasks, workers, verbose, seed_papers, mode):
    """Run a INQUIRO research session."""
    log_file = _setup_logging(verbose)

    click.echo(click.style("\n🔬 INQUIRO Research Engine", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    click.echo(f"  Objective : {objective[:70]}{'...' if len(objective) > 70 else ''}")
    click.echo(f"  Dataset   : {data}")
    click.echo(f"  Cycles    : {cycles}")
    if seed_papers:
        click.echo(f"  Seeds     : {seed_papers}")
    if mode:
        click.echo(f"  Mode      : {mode}")
    click.echo(f"  Log file  : {log_file}")
    click.echo(f"  Log file  : {log_file}")
    click.echo(click.style("=" * 50, fg="cyan"))
    click.echo()

    # Import here to avoid slow startup when just running --help
    from config.settings import settings
    click.echo(f"  Model     : {settings.llm.provider} / {settings.llm.model}\n")

    from src.core.inquiro import Inquiro

    inquiro = Inquiro(
        objective=objective,
        data_path=data,
        max_cycles=cycles,
        num_tasks_per_cycle=tasks,
        max_workers=workers,
        seed_papers_dir=seed_papers,
        mode=mode
    )

    try:
        report_path = inquiro.run()
        click.echo()
        click.echo(click.style("✅ Research complete!", fg="green", bold=True))
        click.echo(f"📄 Report: {report_path}")
    except KeyboardInterrupt:
        click.echo()
        click.echo(click.style("⚠️  Run interrupted by user.", fg="yellow"))
        sys.exit(1)
    except Exception as e:
        click.echo()
        click.echo(click.style(f"❌ Error: {e}", fg="red"))
        logging.exception("Fatal error during run")
        sys.exit(1)


@cli.command()
def version():
    """Show INQUIRO version."""
    click.echo("INQUIRO v0.1.0")


if __name__ == "__main__":
    cli()