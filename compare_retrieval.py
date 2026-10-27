"""Run the four instructor questions through the unchanged retrievers at k=2."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
QUESTIONS = [
    ("Q1", "May a trained graduate student operate the laser alone at 19:00?", "A1; A2"),
    ("Q2", "Can I use the laser without another person present?", "A2"),
    ("Q3", "How frequently must I refresh my laser qualification?", "T1"),
    ("Q4", "What is the laser wavelength?", "absent"),
]
HEADERS = [
    "question_id", "keyword_top_ids", "keyword_scores", "embedding_top_ids",
    "embedding_scores", "required_ids", "keyword_evidence_complete",
    "embedding_evidence_complete", "observation",
]


def run(script, *arguments):
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / script), *arguments, "--k", "2"],
        cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8",
        check=True,
    )
    return result.stdout, result.stderr


def main():
    protected = [ROOT / "keyword_retriever.py", ROOT / "embedding_retriever.py", *sorted((ROOT / "docs").glob("*.txt"))]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    transcript = ["Lesson 10 - Task 3: actual retrieval runs; k=2 for both methods\n"]
    transcript.append("Question and evidence targets supplied by the instructor:\n")
    for qid, question, target in QUESTIONS:
        transcript.append(f"{qid}: {question}\nRequired IDs: {target}\n")
    transcript.append("\nQ4 has no answer in the corpus. Nearest passages and positive scores do not establish answerability.\n")
    keyword = []
    for qid, question, _ in QUESTIONS:
        print(f"Running keyword {qid}", flush=True)
        stdout, stderr = run("keyword_retriever.py", question)
        transcript.append(f"\n=== KEYWORD {qid} (k=2) ===\n{stdout}")
        if stderr:
            transcript.append(f"\nStandard error:\n{stderr}")
        ids = re.findall(r"^  \[([^]]+)\]", stdout, flags=re.MULTILINE)
        scores = dict(re.findall(r"^  (\w+) \| .*? \| score=(\d+) \|", stdout, flags=re.MULTILINE))
        keyword.append((ids, [scores[chunk_id] for chunk_id in ids]))

    print("Running embedding Q1-Q4 on CPU with one shared document encoding", flush=True)
    stdout, stderr = run("embedding_retriever.py", *(question for _, question, _ in QUESTIONS))
    transcript.append(f"\n=== EMBEDDING Q1-Q4 (k=2; documents encoded once) ===\n{stdout}")
    if stderr:
        transcript.append(f"\nEmbedding standard error (model-loading diagnostics):\n{stderr}")
    sections = re.split(r"\nQuestion \d+: ", stdout)[1:]
    assert len(sections) == 4
    rows = []
    for (qid, _, target), (keyword_ids, keyword_scores), section in zip(QUESTIONS, keyword, sections, strict=True):
        selected = re.search(r"Selected IDs \(k=2\): (.+)", section)
        assert selected is not None
        ids = selected.group(1).split(", ")
        scores = dict(re.findall(r"^  (\w+) \| .*? \| dot=([\d.eE+-]+) \|", section, flags=re.MULTILINE))
        assert len(ids) == 2
        rows.append([qid, "; ".join(keyword_ids), "; ".join(keyword_scores), "; ".join(ids), "; ".join(scores[i] for i in ids), target, "", "", ""])

    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected for path, expected in before.items())
    transcript.append("\nOriginal retriever code and source passages verified unchanged by SHA-256.\n")
    (ROOT / "evidence" / "retrieval-runs.txt").write_text("".join(transcript), encoding="utf-8")
    (ROOT / "evidence" / "retrieval-results-data.json").write_text(json.dumps([HEADERS, *rows], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
