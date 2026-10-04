# Erasmus Knowledge Assistant

A retrieval-based knowledge assistant that makes synthetic Erasmus exchange experiences and official information searchable for prospective students at ISEP Paris.

The project is developed as part of the WFP2 course at FH Hochschule Wien.

## Project Goal

The goal is to transform unstructured student experience interviews into a structured, searchable knowledge base. The system will support prospective Erasmus students with questions about preparation, accommodation, courses, ECTS recognition, living costs, and everyday life in Paris.

The current version focuses on preprocessing interview data and implementing keyword-based retrieval. Semantic search and LLM-based answering are planned as later extensions.

## Data

The dataset consists of 15 fully synthetic interview transcripts with fictional Erasmus students at ISEP Paris.

- `data/raw/interviews/` — original interview transcripts (`P01.txt` to `P15.txt`)
- `data/raw/personas/` — persona descriptions
- `data/processed/questions.json` — central interview guide
- `data/processed/personas.json` — structured persona metadata
- `data/processed/interviews.json` — structured interview answers linked to personas and questions

All persons, experiences, and statements are synthetic and must not be interpreted as real student reports.

## Project Structure

```text
data/
├── raw/
│   ├── questions/
│   ├── interviews/
│   └── personas/
└── processed/
    ├── questions.json
    ├── personas.json
    └── interviews.json

scripts/
├── build_interviews.py
└── check_interviews.py
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Build the Interview Dataset

Convert the raw interview transcripts into a structured JSON file:

```bash
python scripts/build_interviews.py
```

Check the generated dataset:

```bash
python scripts/check_interviews.py
```

## Current Status

- [x] Create synthetic interview dataset
- [x] Define persona metadata schema
- [x] Define central interview guide
- [x] Convert interviews into structured JSON
- [ ] Implement BM25 keyword search
- [ ] Evaluate retrieval quality
- [ ] Add semantic search
- [ ] Add LLM-based answer generation

## Tech Stack

- Python
- JSON
- Git / GitHub
- BM25 for keyword retrieval