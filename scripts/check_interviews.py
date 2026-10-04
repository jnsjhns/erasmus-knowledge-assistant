import json
from pathlib import Path

data = json.loads(Path("data/processed/interviews.json").read_text(encoding="utf-8"))

print("Interviews:", len(data["interviews"]))

for interview in data["interviews"]:
    print(interview["interview_id"], interview["persona_id"], len(interview["answers"]))