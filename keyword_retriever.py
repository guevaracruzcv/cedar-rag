"""Lesson 10, Task 1: transparent keyword-overlap retrieval (no AI calls)."""

import argparse
import re
from pathlib import Path


STOPWORDS = set("a an the is are to of in at for and or may can i it be".split())
SOURCE_FILES = ("access.txt", "training.txt", "booking.txt")
LABEL = re.compile(r"^\[([A-Z][0-9]+)\]\s*(.+)$")


def tokenize(text):
    """Return distinct lowercased alphanumeric words, excluding stopwords."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def load_chunks():
    """Read paragraphs from the three source files, keeping source metadata."""
    docs_dir = Path(__file__).resolve().parent / "docs"
    chunks = []
    for filename in SOURCE_FILES:
        for line in (docs_dir / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            match = LABEL.fullmatch(line.strip())
            if match is None:
                raise ValueError(f"Unlabeled paragraph in {filename}: {line!r}")
            chunk_id, paragraph_text = match.groups()
            chunks.append({"id": chunk_id, "file": filename, "text": paragraph_text})
    if len(chunks) != 6 or len({chunk["id"] for chunk in chunks}) != 6:
        raise ValueError("The Cedar Lab source must contain six unique chunks")
    return chunks


def retrieve(question, k):
    """Score and rank all chunks; select only positive scores, up to k."""
    query_words = tokenize(question)
    ranked = []
    for chunk in load_chunks():
        matches = sorted(query_words & tokenize(chunk["text"]))
        ranked.append({**chunk, "matches": matches, "score": len(matches)})
    ranked.sort(key=lambda c: (-c["score"], c["id"]))
    return ranked, [c for c in ranked if c["score"] > 0][:k]


def main():
    parser = argparse.ArgumentParser(description="Cedar Lab keyword retriever")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--k", type=int, default=2, help="Max positive-scoring passages (default: 2)")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    ranked, selected = retrieve(args.question, args.k)
    print(f"Question: {args.question}")
    print(f"Query words after stopword removal: {', '.join(sorted(tokenize(args.question)))}")
    print("\nAll six scores (descending score, ascending ID for ties):")
    for chunk in ranked:
        matches = ", ".join(chunk["matches"]) or "(none)"
        print(f"  {chunk['id']} | {chunk['file']} | score={chunk['score']} | matched: {matches}")

    print(f"\nSelected passages (k={args.k}):")
    if not selected:
        print("No matching passages.")
        return
    for chunk in selected:
        print(f"  [{chunk['id']}] {chunk['file']}: {chunk['text']}")


if __name__ == "__main__":
    main()
