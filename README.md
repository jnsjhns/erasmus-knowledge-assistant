# Erasmus Knowledge Assistant

A retrieval-based knowledge assistant for prospective exchange students at ISEP Paris. The project organizes synthetic student interviews and official information into a searchable knowledge base.

Developed as part of the WFP2 course at FH Campus Wien.

## Current Features

- Structured interview data built from 15 synthetic transcripts
- BM25 keyword search across interview answers and official information
- Search returns up to 20 positive-scoring results by default
- Evaluation on 14 test queries using a graded Silver Standard
- Precision@10, Recall@10, MRR@10, and nDCG@10 metrics

Semantic retrieval and LLM-based answer generation are not implemented yet.

## Data

All people, interviews, and experiences in this project are synthetic. They do not describe real ISEP students.

- `data/raw/interviews/` — source interview transcripts (`P01.txt` to `P15.txt`)
- `data/raw/personas/` — source persona descriptions
- `data/raw/questions/` — interview guide source
- `data/processed/interviews.json` — structured answers linked to interviews and questions
- `data/processed/personas.json` — structured persona metadata
- `data/processed/questions.json` — structured interview guide
- `data/processed/official_info.json` — official information used by search
- `data/evaluation/test_queries.json` — evaluation queries and information needs
- `data/evaluation/silver_standard/` — relevance judgments for each query (`Q01.json` to `Q14.json`)
- `data/evaluation/relevance_score.json` — relevance scale and threshold definitions

## Setup

Create and activate a virtual environment, then install the dependency:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Git Bash on Windows
source .venv/Scripts/activate

# macOS or Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Build and Check the Dataset

Build the structured interview dataset from the raw transcripts, then validate it:

```bash
python scripts/build_interviews.py
python scripts/check_interviews.py
```

## BM25 Search

Run a single query. The default is up to 20 positive-scoring results; `--top-k` can override the limit.

```bash
python scripts/bm25_search.py "wohnung kosten paris" --json
```

Save a query result to a JSON file:

```bash
python scripts/bm25_search.py "wohnung kosten paris" --top-k 20 --output results/wohnung_kosten.json
```

Generate BM25 results for all evaluation queries:

```bash
python scripts/generate_bm25_results.py
```

This writes up to 20 positive-scoring results per query to `results/bm25/` as `bm25_q01_top20.json`, `bm25_q02_top20.json`, and so on.

## Evaluation

The relevance judgments use scores from 0 to 5. Scores of 3 or higher count as relevant for Precision, Recall, and MRR. nDCG uses the graded scores.

Run the metric calculation:

```bash
python scripts/calculate_metrics.py
```

The metric script reads the Top-20 files in `results/bm25/`, evaluates their first 10 results, and writes the report to `results/metrics/bm25_metrics.json`.

Queries with no relevant Silver Standard judgments have undefined Recall. Their per-query Recall is stored as `null` and excluded from the mean Recall.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│       ├── silver_standard/
│       ├── relevance_score.json
│       └── test_queries.json
│
├── results/
│   ├── bm25/                 # Top-20 BM25 result files
│   └── metrics/              # Aggregated evaluation metrics
│
└── scripts/
	├── bm25_search.py
	├── build_interviews.py
	├── calculate_metrics.py
	├── check_interviews.py
	└── generate_bm25_results.py
```

