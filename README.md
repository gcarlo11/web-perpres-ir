# Perpres Hybrid Retrieval & QDA Web Demo

Standalone FastAPI + shadcn-styled dashboard for searching Indonesian Presidential Regulation (Perpres) clauses using `BM25 + intfloat/multilingual-e5-small + RRF + QDA`.

## Included Files (Standalone Repo)
- `main.py` — FastAPI server (`/`, `/health`, `/search`, `/docs`)
- `engine.py` — Two-stage retrieval & QDA reranking engine
- `static/index.html` — Search dashboard UI
- `data/dataset_korpus_per_ayat.csv` — 1,509 Perpres clause segments
- `qda_best.pt` & `config.json` — Trained QDA weights (6.78 MB) and configuration
- `setup_local.py` — Optional one-time offline downloader for `intfloat/multilingual-e5-small` and `artifacts/doc_artifacts.pt`
- `requirements.txt` — Python dependencies

*(Note: `.venv/`, `models/`, and `artifacts/doc_artifacts.pt` (~285 MB) are excluded in `.gitignore` so the repo can be pushed cleanly to GitHub without exceeding the 100 MB file limit. On a fresh clone, `doc_artifacts.pt` is automatically generated on first startup or via `python setup_local.py`.)*

## Setup & Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python setup_local.py   # optional: downloads base encoder & pre-builds doc_artifacts.pt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the web dashboard, or `http://localhost:8000/docs` for Swagger API docs.
