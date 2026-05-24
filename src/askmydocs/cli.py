"""CLI entrypoint. Grows one command per phase as we add features."""

from __future__ import annotations  # noqa: I001

import typer

from askmydocs.chunk import chunk_sections
from askmydocs.ingest import ingest_all
from askmydocs.embed import build_index, search
from askmydocs.generate import answer_question

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root() -> None:
    """Ask My Docs — RAG over technical PDFs."""
    # Forces Typer into multi-command mode so subcommands like `ingest` aren't auto-promoted.


@app.command()
def ingest() -> None:
    """Parse every PDF in data/raw/ and write data/processed/pages.jsonl."""
    ingest_all()

@app.command()
def chunk() -> None:
    """Split data/processed/sections.jsonl into data/processed/chunks.jsonl."""
    chunk_sections()

@app.command()
def embed() -> None:
    """Embed data/processed/chunks.jsonl into the Chroma vector store."""
    build_index()


@app.command()
def query(text: str, k: int = 5) -> None:
    """Semantic search over the indexed papers (retrieval only, no LLM yet)."""
    from rich.console import Console
    console = Console()
    for i, (chunk, score) in enumerate(search(text, k=k), start=1):
        console.print(f"\n[bold]{i}. {chunk.source.display()}[/bold]  [dim]dist={score:.3f}[/dim]")
        console.print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))

@app.command()
def ask(question: str, k: int = 5) -> None:
    """Ask a question; get a grounded answer with citations."""
    from rich.console import Console
    console = Console()
    ans = answer_question(question, k=k)

    console.print(f"\n[bold]{ans.text}[/bold]\n")
    if ans.cited:
        console.print("[dim]Sources:[/dim]")
        for n, chunk in ans.cited:
            console.print(f"  [{n}] {chunk.source.display()}")
    else:
        console.print("[dim](no sources cited)[/dim]")

if __name__ == "__main__":
    app()
