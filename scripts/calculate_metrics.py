import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SILVER_STANDARD_DIR = ROOT / "data" / "evaluation" / "silver_standard"
BM25_RESULTS_DIR = ROOT / "results" / "bm25"
METRICS_FILE = ROOT / "results" / "metrics" / "bm25_metrics.json"

K = 10
RELEVANT_THRESHOLD = 3


def load_json(path: Path) -> dict:
    """Lädt eine JSON-Datei mit UTF-8-Encoding."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_dcg(relevance_scores: list[int]) -> float:
    """
    Berechnet DCG@K mit exponentiellem Gain:
    gain = 2^relevance - 1.
    """
    return sum(
        (2 ** relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevance_scores[:K], start=1)
    )


def get_document_ids(search_result: dict) -> list[str]:
    """
    Liest die BM25-Dokument-IDs aus dem Feld 'results'.
    Es werden maximal die ersten K Ergebnisse verwendet.
    """
    return [
        result["document_id"]
        for result in search_result.get("results", [])[:K]
    ]


def calculate_metrics(silver_standard: dict, bm25_result: dict) -> dict:
    """
    Vergleicht ein BM25-Top-K-Ranking mit dem Silver Standard einer Query.
    Nicht im Silver Standard gespeicherte Dokumente erhalten implizit
    den Relevanzscore 0.
    """
    judgments = {
        item["document_id"]: item["relevance_score"]
        for item in silver_standard.get("results", [])
    }

    retrieved_document_ids = get_document_ids(bm25_result)

    retrieved_relevance_scores = [
        judgments.get(document_id, 0)
        for document_id in retrieved_document_ids
    ]

    # Für echte @10-Metriken: Falls BM25 weniger als K Treffer liefert,
    # fehlende Plätze als nicht relevant bzw. Relevanzscore 0 behandeln.
    retrieved_relevance_scores.extend(
        [0] * (K - len(retrieved_relevance_scores))
    )

    relevant_document_ids = {
        document_id
        for document_id, relevance_score in judgments.items()
        if relevance_score >= RELEVANT_THRESHOLD
    }

    retrieved_relevant_document_ids = {
        document_id
        for document_id, relevance_score in zip(
            retrieved_document_ids,
            retrieved_relevance_scores
        )
        if relevance_score >= RELEVANT_THRESHOLD
    }

    relevant_retrieved_count = len(retrieved_relevant_document_ids)

    precision_at_10 = relevant_retrieved_count / K

    recall_at_10 = (
        relevant_retrieved_count / len(relevant_document_ids)
        if relevant_document_ids
        else None
    )

    first_relevant_rank = next(
        (
            rank
            for rank, relevance_score in enumerate(
                retrieved_relevance_scores,
                start=1
            )
            if relevance_score >= RELEVANT_THRESHOLD
        ),
        None,
    )

    mrr_at_10 = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    dcg_at_10 = calculate_dcg(retrieved_relevance_scores)

    ideal_relevance_scores = sorted(
        judgments.values(),
        reverse=True
    )[:K]

    ideal_relevance_scores.extend(
        [0] * (K - len(ideal_relevance_scores))
    )

    idcg_at_10 = calculate_dcg(ideal_relevance_scores)

    ndcg_at_10 = (
        dcg_at_10 / idcg_at_10
        if idcg_at_10 > 0
        else 0.0
    )

    return {
        "query_id": silver_standard["query_id"],
        "query": silver_standard["query"],
        "num_retrieved": len(retrieved_document_ids),
        "num_relevant_in_corpus": len(relevant_document_ids),
        "relevant_retrieved_at_10": relevant_retrieved_count,
        "retrieved_document_ids_at_10": retrieved_document_ids,
        "retrieved_relevance_scores_at_10": retrieved_relevance_scores,
        "precision_at_10": precision_at_10,
        "recall_at_10": recall_at_10,
        "mrr_at_10": mrr_at_10,
        "dcg_at_10": dcg_at_10,
        "idcg_at_10": idcg_at_10,
        "ndcg_at_10": ndcg_at_10,
    }


def main():
    if not SILVER_STANDARD_DIR.exists():
        raise FileNotFoundError(
            f"Silver-Standard-Ordner nicht gefunden: {SILVER_STANDARD_DIR}"
        )

    if not BM25_RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"BM25-Ergebnisordner nicht gefunden: {BM25_RESULTS_DIR}"
        )

    silver_files = sorted(SILVER_STANDARD_DIR.glob("Q*.json"))

    if not silver_files:
        raise FileNotFoundError(
            f"Keine Silver-Standard-Dateien in {SILVER_STANDARD_DIR} gefunden."
        )

    all_metrics = []
    missing_bm25_results = []

    for silver_file in silver_files:
        silver_standard = load_json(silver_file)
        query_id = silver_standard["query_id"]

        # Q01 -> bm25_q01_top20.json
        bm25_file = BM25_RESULTS_DIR / f"bm25_{query_id.lower()}_top20.json"

        if not bm25_file.exists():
            print(f"Kein BM25-Ergebnis für {query_id}: {bm25_file.name}")
            missing_bm25_results.append(query_id)
            continue

        bm25_result = load_json(bm25_file)

        metrics = calculate_metrics(silver_standard, bm25_result)
        all_metrics.append(metrics)

        recall_display = (
            f"{metrics['recall_at_10']:.4f}"
            if metrics["recall_at_10"] is not None
            else "n/a"
        )
        print(
            f"{query_id}: "
            f"nDCG@10={metrics['ndcg_at_10']:.4f}, "
            f"P@10={metrics['precision_at_10']:.4f}, "
            f"R@10={recall_display}, "
            f"MRR@10={metrics['mrr_at_10']:.4f}"
        )

    if not all_metrics:
        raise ValueError(
            "Es konnten keine BM25-Ergebnisse gegen den Silver Standard ausgewertet werden."
        )

    mean_ndcg_at_10 = sum(
        item["ndcg_at_10"] for item in all_metrics
    ) / len(all_metrics)

    mean_precision_at_10 = sum(
        item["precision_at_10"] for item in all_metrics
    ) / len(all_metrics)

    defined_recalls = [
        item["recall_at_10"]
        for item in all_metrics
        if item["recall_at_10"] is not None
    ]
    mean_recall_at_10 = (
        sum(defined_recalls) / len(defined_recalls)
        if defined_recalls
        else None
    )

    mean_mrr_at_10 = sum(
        item["mrr_at_10"] for item in all_metrics
    ) / len(all_metrics)

    summary = {
        "retriever": "bm25",
        "k": K,
        "relevant_threshold": RELEVANT_THRESHOLD,
        "relevance_scale": "0-5",
        "num_queries_evaluated": len(all_metrics),
        "num_queries_with_defined_recall": len(defined_recalls),
        "missing_bm25_results": missing_bm25_results,
        "mean_ndcg_at_10": round(mean_ndcg_at_10, 4),
        "mean_precision_at_10": round(mean_precision_at_10, 4),
        "mean_recall_at_10": (
            round(mean_recall_at_10, 4)
            if mean_recall_at_10 is not None
            else None
        ),
        "mean_mrr_at_10": round(mean_mrr_at_10, 4),
        "per_query": [
            {
                **item,
                "precision_at_10": round(item["precision_at_10"], 4),
                "recall_at_10": (
                    round(item["recall_at_10"], 4)
                    if item["recall_at_10"] is not None
                    else None
                ),
                "mrr_at_10": round(item["mrr_at_10"], 4),
                "dcg_at_10": round(item["dcg_at_10"], 4),
                "idcg_at_10": round(item["idcg_at_10"], 4),
                "ndcg_at_10": round(item["ndcg_at_10"], 4),
            }
            for item in all_metrics
        ],
    }

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nGesamtergebnis:")
    print(f"Ausgewertete Queries: {len(all_metrics)}")
    print(f"Mean nDCG@10:      {mean_ndcg_at_10:.4f}")
    print(f"Mean Precision@10: {mean_precision_at_10:.4f}")
    mean_recall_display = (
        f"{mean_recall_at_10:.4f}"
        if mean_recall_at_10 is not None
        else "n/a"
    )
    print(f"Mean Recall@10:    {mean_recall_display}")
    print(f"Mean MRR@10:       {mean_mrr_at_10:.4f}")
    print(f"Gespeichert: {METRICS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()