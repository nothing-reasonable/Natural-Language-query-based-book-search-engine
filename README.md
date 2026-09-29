# বইখোঁজ

Bangla-first book discovery over a local catalogue. The application keeps the existing
Flask/Python hybrid search stack: exact title, BM25, multilingual dense retrieval,
metadata facets, knowledge graph, reciprocal-rank fusion, and optional reranking.

## Setup on the GPU PC

Use Python 3.11 or newer. Create and activate a virtual environment, then install the
PyTorch wheel appropriate for that PC before the remaining requirements.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `BOOKSEARCH_EMBEDDING_MODEL_REVISION` to a fixed model tag or commit for the demo
build. The manifest records the model identity, revision, query/document prompts,
dimension, catalogue fingerprint, text-preparation version, and passage settings; equal
dimensions alone never make two indexes compatible.

## Migrate and rebuild

Keep a copy of the current `artifacts` directory before migration. Existing book IDs are
unchanged. Legacy enrichment rows without an input fingerprint are treated as stale:
search uses deterministic dictionary tags until enrichment is regenerated.

```powershell
python cli.py ingest books_metadata_cleaned.csv
python cli.py enrich                 # or: python cli.py enrich --no-llm
python cli.py build-index
python cli.py doctor
```

The dense index stores 256-token passages with 48-token overlap from descriptions and
available tables of contents. Builds resume unchanged rows, update changed fingerprints,
delete obsolete rows, and publish `artifacts/index_manifest.json` only after all requested
artifacts succeed. If a build is interrupted, startup refuses the incomplete generation
and prints the rebuild command.

To roll back, stop the server, restore the complete backed-up `artifacts` directory
(including its manifest), restore the matching application revision, and start again.
Never mix one backup's manifest with another backup's indexes.

## Run

```powershell
python cli.py serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Public browser responses exclude paths, raw exceptions,
profile identifiers, model controls, and ranking scores. To enable research-only browser
diagnostics on a trusted machine, set `BOOKSEARCH_WEB_DIAGNOSTICS=true`; CLI diagnostics
remain available regardless.

Useful endpoints:

- `POST /api/search` — query, structured filters, stable pagination;
- `GET /api/catalog` — exact filtered catalogue counts and title/year ordering;
- `GET /api/facets` — catalogue-backed filter values;
- `GET /api/books/<book_id>` — shareable book details;
- `GET /api/status` — `loading`, `ready`, `degraded`, or `failed` operation.

## Verification on the other PC

Do not download models or rebuild indexes on a development-only machine. On the target
PC, after the rebuild:

```powershell
pytest -q
python evaluation.py migrate --outputs outputs --rank-judgments judgments.yaml --out eval/judgments_book_ids.json
python evaluation.py run --queries eval/queries.json --out eval/updated-results.json
python evaluation.py score --results eval/updated-results.json --judgments eval/judgments_book_ids.json --out eval/updated-report.json
python evaluation.py score --results eval/updated-results.json --baseline eval/baseline-results.json --judgments eval/judgments_book_ids.json --out eval/comparison.json
```

The migration writes only unambiguous title/author-to-book-ID mappings and lists every
unresolved rank for review; it never edits `judgments.yaml` or existing `outputs/`.
Evaluation JSON reports nDCG@10, MRR@10, candidate pooled Recall@60, judgment coverage,
and warm median/p95 latency overall and separately for Bangla, English, exact-title,
filtered, and semantic queries. Ranking gains remain unverified until these commands run
against the same catalogue on the target PC. Filter correctness and failure-path tests
must pass before demo acceptance.
