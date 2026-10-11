# scripts/generate_bm25_results.py

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = ROOT / "data" / "evaluation"
TEST_QUERIES_FILE = EVALUATION_DIR / "test_queries.json"
RESULTS_DIR = ROOT / "results" / "bm25"

BM25_SCRIPT = ROOT / "scripts" / "bm25_search.py"


def run_bm25_search(query: str, top_k: int = 20) -> dict:
    """Führt das bestehende BM25-Skript aus und liefert die JSON-Ausgabe."""

    command = [
        sys.executable,
        str(BM25_SCRIPT),
        query,
        "--top-k",
        str(top_k),
        "--json",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=True,
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Das BM25-Skript hat keine gültige JSON-Ausgabe geliefert.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ) from error


def main():
    if not TEST_QUERIES_FILE.exists():
        raise FileNotFoundError(
            f"Testfragen nicht gefunden: {TEST_QUERIES_FILE}"
        )

    if not BM25_SCRIPT.exists():
        raise FileNotFoundError(
            f"BM25-Skript nicht gefunden: {BM25_SCRIPT}"
        )

    with TEST_QUERIES_FILE.open("r", encoding="utf-8") as file:
        evaluation_data = json.load(file)

    test_queries = evaluation_data.get("test_queries", [])

    if not test_queries:
        raise ValueError(
            "In test_queries.json wurden keine Testfragen gefunden."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starte BM25-Evaluation für {len(test_queries)} Testfragen.\n")

    for test_query in test_queries:
        query_id = test_query["query_id"]
        query = test_query["query"]

        print(f"[{query_id}] {query}")

        try:
            search_result = run_bm25_search(query, top_k=20)
        except Exception as error:
            print(f"  Fehler: {error}\n")
            continue

        output = {
            "query_id": query_id,
            "query": query,
            "information_need": test_query.get("information_need"),
            "retriever": "bm25",
            "top_k": 20,
            "results": search_result.get("results", []),
        }

        output_filename = f"bm25_{query_id.lower()}_top20.json"
        output_path = RESULTS_DIR / output_filename

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        print(f"  Gespeichert: {output_path.relative_to(ROOT)}\n")

    print("BM25-Evaluation abgeschlossen.")


if __name__ == "__main__":
    main()