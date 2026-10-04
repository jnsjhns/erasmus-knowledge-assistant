import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "interviews"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "questions.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "interviews.json"

with QUESTIONS_FILE.open(encoding="utf-8") as f:
    questions_data = json.load(f)

guide = questions_data["interview_guide"]

standard_questions = {
    question["number"]: question["question_id"]
    for question in guide["questions"]
}

question_pattern = re.compile(r"^(\d+(?:\.\d+)?)\.\s+(.*)$")


def parse_interview(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()

    segments = []
    current = None

    for line in lines:
        line = line.strip()

        if not line:
            if current is not None:
                current["text"] += "\n"
            continue

        match = question_pattern.match(line)

        if match:
            if current is not None:
                segments.append(current)

            number_text, question_text = match.groups()
            number = float(number_text)

            is_standard = number in standard_questions

            current = {
                "number": number_text,
                "question_id": standard_questions.get(number),
                "question_text": None if is_standard else question_text,
                "parent_question_id": None,
                "text": ""
            }

            if "." in number_text:
                parent_number = float(number_text.split(".")[0])
                current["parent_question_id"] = standard_questions.get(parent_number)

        elif current is not None:
            if current["text"]:
                current["text"] += " "
            current["text"] += line

    if current is not None:
        segments.append(current)

    for segment in segments:
        segment["text"] = segment["text"].strip()
        segment["answer_id"] = (
            f"{path.stem.lower()}_q{segment['number'].replace('.', '_')}"
        )

    return segments


interviews = []

for path in sorted(RAW_DIR.glob("P*.txt"), key=lambda p: int(re.search(r"\d+", p.stem).group())):
    match = re.search(r"P(\d+)", path.stem, re.IGNORECASE)

    if not match:
        print(f"Übersprungen, keine Nummer erkannt: {path.name}")
        continue

    persona_number = int(match.group(1))
    answers = parse_interview(path)

    interviews.append({
        "interview_id": f"INT_{persona_number:02d}",
        "persona_id": f"p{persona_number:02d}",
        "guide_id": guide["guide_id"],
        "source_file": path.name,
        "language": guide["language"],
        "answers": answers
    })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump({"interviews": interviews}, f, ensure_ascii=False, indent=2)

print(f"{len(interviews)} Interviews verarbeitet.")
print(f"Gespeichert in: {OUTPUT_FILE}")