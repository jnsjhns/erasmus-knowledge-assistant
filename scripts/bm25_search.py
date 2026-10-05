import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]

INTERVIEWS_FILE = ROOT / "data" / "processed" / "interviews.json"
OFFICIAL_FILE = ROOT / "data" / "processed" / "official_info.json"


STOPWORDS = {
    "der", "die", "das", "und", "oder", "aber", "ein", "eine", "einen",
    "einem", "einer", "ich", "du", "er", "sie", "es", "wir", "ihr",
    "ist", "sind", "war", "waren", "hat", "haben", "wird", "werden",
    "mit", "von", "zu", "zum", "zur", "im", "in", "am", "an", "auf",
    "für", "nicht", "auch", "wie", "was", "wenn", "dann", "sich"
}


def tokenize(text):
    tokens = re.findall(r"[a-zäöüß0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def load_documents():
    documents = []

    if not INTERVIEWS_FILE.exists():
        raise FileNotFoundError(f"Interviews nicht gefunden: {INTERVIEWS_FILE}")

    if not OFFICIAL_FILE.exists():
        raise FileNotFoundError(f"Offizielle Informationen nicht gefunden: {OFFICIAL_FILE}")

    interviews = json.loads(
        INTERVIEWS_FILE.read_text(encoding="utf-8")
    )["interviews"]

    for interview in interviews:
        for answer in interview["answers"]:
            documents.append({
                "source_type": "interview",
                "document_id": answer["answer_id"],
                "interview_id": interview["interview_id"],
                "persona_id": interview["persona_id"],
                "question_id": answer["question_id"],
                "answer_id": answer["answer_id"],
                "topic": "",
                "text": answer["text"]
            })

    official = json.loads(
        OFFICIAL_FILE.read_text(encoding="utf-8")
    )["official_information"]

    for fact in official:
        documents.append({
            "source_type": fact["source_type"],
            "document_id": fact["fact_id"],
            "fact_id": fact["fact_id"],
            "source": fact["source"],
            "topic": fact["topic"],
            "text": fact["text"]
        })

    return documents


def search(query, documents, top_k=5):
    tokenized_corpus = [
        tokenize(document["text"])
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    def get_boost(document):
        if document["source_type"] == "official":
            return 1.2
        return 1.0

    ranked = []

    for score, document in zip(scores, documents):
        final_score = float(score) * get_boost(document)

        if final_score <= 0:
            continue

        ranked.append({
            "document": document,
            "base_score": float(score),
            "score": final_score
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    results = []

    for rank, item in enumerate(ranked[:top_k], start=1):
        document = item["document"]

        results.append({
            "rank": rank,
            "score": round(item["score"], 4),
            "base_score": round(item["base_score"], 4),
            "document_id": document["document_id"],
            "source_type": document["source_type"],
            "topic": document.get("topic", ""),
            "text": document["text"]
        })

    return results


def print_human_results(query, results):
    print(f"\nSuche: {query}\n")
    print(f"Top {len(results)} Treffer:\n")

    if not results:
        print("Keine Treffer gefunden.")
        return

    for result in results:
        document = result
        print(f"{result['rank']}. Score: {result['score']:.2f}")
        print(f"Quelle: {result['source_type']}")

        if result["source_type"] == "interview":
            print(f"Interview: {result['document_id']}")
        else:
            print(f"Fakt: {result['document_id']} | Thema: {result['topic']}")

        print(result["text"][:300] + "...\n")


def main():
    parser = argparse.ArgumentParser(
        description="BM25-Suche über ISEP-Interviews und offizielle Fakten"
    )

    parser.add_argument("query", help="Suchanfrage")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Anzahl der zurückgegebenen Treffer"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Ergebnis als JSON ausgeben"
    )

    args = parser.parse_args()

    documents = load_documents()
    results = search(args.query, documents, top_k=args.top_k)

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "top_k": args.top_k,
                    "results": results
                },
                ensure_ascii=False,
                indent=2
            )
        )
    else:
        print_human_results(args.query, results)


if __name__ == "__main__":
    main()