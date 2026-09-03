"""Command line interface.

    python -m booksearch doctor
    python -m booksearch ingest books_metadata_cleaned.csv
    python -m booksearch enrich
    python -m booksearch build-index
    python -m booksearch search "পাকিস্তান আমলে কূটনীতিক ছিলেন এমন লেখকদের মুক্তিযুদ্ধের বই"
    python -m booksearch serve
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from config import settings
from search.indexing.embedding import make_embedder
from search.indexing.kg_index import KnowledgeGraph
from search.indexing.bm25_index import LexicalIndex
from search.ranking.profile_index import ProfileStore
from search.indexing.dense_index import VectorIndex
from ingest import run as ingest_run
from search.llm import LMStudio
from search.engine import SearchEngine

app = typer.Typer(add_completion=False, help="বাংলা বই অনুসন্ধান / Bengali book search engine")
console = Console(record=True, force_terminal=False)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def doctor():
    """Check LM Studio connectivity and which artifacts exist."""
    llm = LMStudio(settings)
    table = Table("check", "status")
    try:
        models = llm.list_models()
        table.add_row("LM Studio", f"[green]ok[/] — {settings.lmstudio_base_url}")
        table.add_row("models", ", ".join(models) or "[yellow]none loaded[/]")
        for label, getter in (("chat model", lambda: llm.chat_model),
                              ("embedding model", lambda: llm.embedding_model)):
            try:
                table.add_row(label, getter())
            except Exception as exc:  # noqa: BLE001
                table.add_row(label, f"[yellow]{exc}[/]")
    except Exception as exc:  # noqa: BLE001
        table.add_row("LM Studio", f"[red]{exc}[/]")

    table.add_row("embedding backend", settings.embedding_backend)
    try:
        embedder = make_embedder(settings)
        table.add_row("embedding model", f"{embedder.name} ({embedder.dimension}-dim)")
    except Exception as exc:  # noqa: BLE001
        table.add_row("embedding model", f"[red]{exc}[/]")

    for label, path in (
        ("books.jsonl", settings.books_path),
        ("enrichment.jsonl", settings.enrichment_path),
        ("lexical index", settings.lexical_dir),
        ("vector index", settings.vector_dir),
        ("knowledge graph", settings.graph_path),
    ):
        table.add_row(label, "[green]present[/]" if path.exists() else "[yellow]missing[/]")
    console.print(table)


@app.command()
def ingest(source: Path = typer.Argument(..., help="CSV exported by the crawler")):
    """Load, normalise, de-duplicate and resolve author identities."""
    ingest_run.ingest(source, settings)


@app.command()
def enrich(
    no_llm: bool = typer.Option(False, "--no-llm", help="dictionary tagging only"),
    limit: int = typer.Option(0, help="only process the first N remaining books"),
    redo: bool = typer.Option(False, "--redo", help="discard existing enrichment and start over"),
):
    """Extract subjects, periods, places and author roles. Resumable."""
    ingest_run.enrich(settings, use_llm=not no_llm, limit=limit or None, redo=redo)


@app.command("build-index")
def build_index(
    no_vector: bool = typer.Option(False, "--no-vector", help="skip the dense index (no embedding model)"),
    redo_vector: bool = typer.Option(False, "--redo-vector", help="re-embed everything from scratch"),
):
    """Build the lexical, dense and knowledge-graph indexes."""
    records = ingest_run.load_indexed(settings)
    console.print(f"indexing {len(records)} books")

    LexicalIndex.build(records, settings)
    console.print("[green]lexical[/] index built")

    graph = KnowledgeGraph.build(records, settings)
    console.print(f"[green]graph[/] built — {graph.stats()}")

    if no_vector:
        console.print("[yellow]skipped[/] dense index")
        return
    embedder = make_embedder(settings)
    console.print(f"embedding with [bold]{embedder.name}[/] ({embedder.dimension}-dim)")
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("embedding", total=len(records))
        VectorIndex.build(records, embedder, settings,
                          on_batch=lambda n: progress.advance(task, n), resume=not redo_vector)
    console.print("[green]vector[/] index built")


@app.command()
def search(
    query: str = typer.Argument(..., help="প্রশ্ন (Bengali)"),
    user: str = typer.Option("", help="user id, enables personalisation"),
    top_k: int = typer.Option(0, help="number of results"),
    top: int = typer.Option(15, "--top", help="RAG-Fusion: how many fused results to keep (N)"),
    rag_fusion: bool = typer.Option(False, "--rag-fusion",
                                    help="retrieve for several reformulations of the query and fuse the rankings"),
    no_llm: bool = typer.Option(False, "--no-llm", help="rules-only, no LM Studio at all"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="skip second-stage reranking (faster, lower precision)"),
    plan: bool = typer.Option(False, "--plan", help="use the LLM for queries the rules cannot read"),
    force_plan: bool = typer.Option(False, "--force-plan", help="use the LLM on every query, regardless of the rules"),
    no_plan: bool = typer.Option(False, "--no-plan", help="rules only, never call the LLM"),
    verbose: bool = typer.Option(False, "--verbose", help="show the query plan and score breakdown"),
    show_rerank: bool = typer.Option(False, "--show-rerank",
                                     help="show what the reranker was shown and what it scored"),
    show_passages: bool = typer.Option(False, "--show-passages",
                                       help="with --show-rerank, print each passage in full"),
):
    """Run a search."""
    if sum([plan, force_plan, no_plan]) > 1:
        raise typer.BadParameter("choose one of --plan, --force-plan, --no-plan")
    mode = settings.llm_query_understanding
    if force_plan:
        mode = "always"
    elif plan:
        mode = "auto"
    elif no_plan:
        mode = "never"

    if rag_fusion and no_llm:
        raise typer.BadParameter("--rag-fusion needs LM Studio to write the query variants")

    tuned = settings.model_copy(update={
        "use_reranker": settings.use_reranker and not no_rerank,
        "llm_query_understanding": "never" if no_llm else mode,
        "rag_fusion": rag_fusion or settings.rag_fusion,
        "rag_fusion_top_n": top,
    })
    if show_passages and not show_rerank:
        raise typer.BadParameter("--show-passages only means something with --show-rerank")

    engine = SearchEngine.load(tuned, use_llm=not no_llm)
    # --top is the RAG-Fusion N; --top-k still names the plain-search result count, so
    # whichever path is running gets the number that belongs to it.
    wanted = top_k or (top if tuned.rag_fusion else None)
    response = engine.search(query, user_id=user or None, top_k=wanted,
                             trace_rerank=show_rerank)

    if verbose:
        # Not `plan` -- that name is the --plan flag in this scope.
        query_plan = response.plan
        consulted = response.timings_ms.get("understand", 0) > 1000
        entities = ", ".join(
            f"{e.name} ({e.kind}{', hard' if e.hard else ''})" for e in query_plan.entities
        ) or "-"
        variant_lines = "".join(f"   • {v}\n" for v in response.query_variants)
        variants_block = (
            f"queries searched ({len(response.query_variants)}):\n{variant_lines}"
            if response.query_variants else ""
        )
        console.print(Panel(
            f"intent: [bold]{query_plan.intent}[/]\n"
            f"query understanding: {tuned.llm_query_understanding}"
            f" — {'LLM consulted' if consulted else 'rules only'}\n"
            f"entities: {entities}\n"
            f"keywords: {', '.join(query_plan.keywords)}\n"
            f"expanded: {', '.join(query_plan.expanded_terms[:12])}\n"
            f"{variants_block}"
            f"filters: {query_plan.filters.model_dump(exclude_defaults=True)}\n"
            f"steps: {[s.model_dump(exclude_defaults=True) for s in query_plan.steps]}\n"
            f"timings(ms): {response.timings_ms}",
            title="query plan",
        ))

    if show_rerank:
        _print_rerank(response.rerank, show_passages)

    if not response.hits:
        console.print("[yellow]কোনো ফলাফল পাওয়া যায়নি।[/]")
        return

    for rank, hit in enumerate(response.hits, start=1):
        header = f"[bold]{rank}. {hit.book.title}[/] — {hit.book.author}"
        body = [hit.explanation]
        if verbose:
            body.append(f"[dim]score {hit.score} | {hit.components} | channels: {', '.join(hit.channels)}[/]")
        console.print(Panel("\n".join(body), title=header, title_align="left"))



def _print_rerank(trace, show_passages: bool) -> None:
    """Print reranker input and output, in the order the reranker put them."""
    if trace is None or not trace.entries:
        console.print("[yellow]reranker produced no trace[/] — nothing reached stage 2")
        return

    # `noop` is what every backend degrades to on a failed load or a dead server, and its
    # evenly-spaced scores read exactly like a real reranking. Say so plainly.
    if trace.backend == "noop":
        console.print(Panel(
            "The reranker is [bold]not[/] running: scores below are fusion order "
            "restated as a linear ramp. Check for a 'cross-encoder unavailable' "
            "warning to see why.",
            title="[yellow]no reranking[/]", title_align="left",
        ))

    console.print(Panel(
        f"""backend: [bold]{trace.backend}[/]
model: {trace.model or ('(whatever LM Studio has loaded)' if trace.backend == 'lmstudio' else '—')}
query as fed to the reranker: {trace.query}
candidates scored: {len(trace.entries)}""",
        title="reranker", title_align="left",
    ))

    table = Table("in", "score", "final", "moved", "title")
    for entry in sorted(trace.entries, key=lambda e: -e.score):
        if entry.final_rank is None:
            movement = "[dim]dropped[/]"
        else:
            delta = entry.fusion_rank - entry.final_rank
            movement = (f"[green]+{delta}[/]" if delta > 0
                        else f"[red]{delta}[/]" if delta < 0 else "[dim]0[/]")
        table.add_row(str(entry.fusion_rank), f"{entry.score:.4f}",
                      str(entry.final_rank or "-"), movement, entry.title[:60])
    console.print(table)

    if show_passages:
        for entry in sorted(trace.entries, key=lambda e: -e.score):
            console.print(Panel(entry.passage,
                                title=f"in #{entry.fusion_rank} — score {entry.score:.4f}",
                                title_align="left"))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="bind address"),
    port: int = typer.Option(8000, help="port"),
    no_llm: bool = typer.Option(False, "--no-llm", help="rules-only, no LM Studio at all"),
):
    """Serve the web UI. The engine loads in the background; the page shows progress."""
    import web

    console.print(f"[green]serving[/] http://{host}:{port} — first load takes ~30s")
    web.serve(settings, host=host, port=port, use_llm=not no_llm)


@app.command()
def profile(
    user: str = typer.Argument(...),
    like_genre: list[str] = typer.Option([], "--genre", help="explicit genre preference"),
    like_subject: list[str] = typer.Option([], "--subject", help="explicit subject preference"),
    click: list[str] = typer.Option([], "--click", help="record a click on a book_id"),
    save_book: list[str] = typer.Option([], "--save", help="record a saved book_id"),
    show: bool = typer.Option(False, "--show", help="print the stored profile"),
):
    """Inspect or update a user profile."""
    store = ProfileStore(settings)
    record = store.get(user)
    record.genres = list(dict.fromkeys(record.genres + list(like_genre)))
    record.subjects = list(dict.fromkeys(record.subjects + list(like_subject)))
    for book_id in click:
        record.record("click", book_id)
    for book_id in save_book:
        record.record("save", book_id)
    store.save(record)

    updates = list(like_genre) + list(like_subject) + list(click) + list(save_book)
    if updates:
        # Saving without saying so reads as a no-op; the file is easy to miss.
        console.print(
            f"[green]saved[/] {settings.profiles_dir / (user + '.json')} — "
            f"{len(record.genres)} genres, {len(record.subjects)} subjects, "
            f"{len(record.clicks)} clicked, {len(record.saved)} saved"
        )
    if show or not updates:
        console.print_json(record.model_dump_json(indent=2))


if __name__ == "__main__":
    app()

