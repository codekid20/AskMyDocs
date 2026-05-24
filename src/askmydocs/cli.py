"""CLI entrypoint. Grows one command per phase as we add features."""

from __future__ import annotations

import typer

from askmydocs.ingest import ingest_all

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root() -> None:
    """Ask My Docs — RAG over technical PDFs."""
    # Forces Typer into multi-command mode so subcommands like `ingest` aren't auto-promoted.


@app.command()
def ingest() -> None:
    """Parse every PDF in data/raw/ and write data/processed/pages.jsonl."""
    ingest_all()


if __name__ == "__main__":
    app()
