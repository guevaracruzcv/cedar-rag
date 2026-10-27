"""Lesson 10, Task 2: inspect real CPU embeddings and retrieve Cedar passages."""

import argparse
from importlib.metadata import version

import numpy as np
from sentence_transformers import SentenceTransformer

from keyword_retriever import load_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
Q1 = "May a trained graduate student operate the laser alone at 19:00?"


def coordinates(vector):
    return " ".join(f"{value:.4f}" for value in vector[:12])


def rank_passages(documents, question, chunks):
    """Use every coordinate; promote model values to float64 for arithmetic."""
    docs = documents.astype(np.float64)
    query = question.astype(np.float64)
    scores = docs @ query
    lengths = np.linalg.norm(docs, axis=1) * np.linalg.norm(query)
    cosines = scores / lengths
    np.testing.assert_allclose(scores, cosines, rtol=1e-6, atol=1e-7)
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["id"]))
    return order, scores, cosines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="Quoted questions; defaults to Q1")
    parser.add_argument("--k", type=int, default=2, help="Number of passages (default: 2)")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    print(f"Model: {MODEL_NAME}", flush=True)
    for package in ("numpy", "sentence-transformers", "torch", "transformers", "huggingface-hub"):
        print(f"Package {package}: {version(package)}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"Device: {model.device}")
    chunks = load_chunks()
    # Only passage text reaches the model. IDs and filenames stay in chunks.
    documents = model.encode_document(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if documents.shape != (6, 384):
        raise ValueError(f"Unexpected document matrix shape: {documents.shape}")
    print(f"Document matrix shape: {documents.shape}; dtype: {documents.dtype}")
    print("Document embeddings computed once and reused across questions.")
    print("All 384 coordinates, not just the displayed 12, are used in retrieval.")
    print("Coordinates below are rounded only for display; arithmetic uses unrounded values.")
    print("Model float32 values are promoted to float64 for scoring and inspection arithmetic.")
    doc_lengths = np.linalg.norm(documents.astype(np.float64), axis=1)
    np.testing.assert_allclose(doc_lengths, 1.0, rtol=1e-6, atol=1e-7)
    displayed = []
    print("\nDocument vectors (first 12 coordinates):")
    for i, chunk in enumerate(chunks):
        print(f"  {chunk['id']} | {chunk['file']} | {coordinates(documents[i])}")
        print(f"    Vector length: {doc_lengths[i]:.9f}; approximately 1: True")
        displayed.extend((float(value), chunk["id"], j + 1) for j, value in enumerate(documents[i, :12]))

    for number, question in enumerate(args.questions or [Q1], start=1):
        query = model.encode_query(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if query.shape != (384,):
            raise ValueError(f"Unexpected question vector shape: {query.shape}")
        query_length = np.linalg.norm(query.astype(np.float64))
        np.testing.assert_allclose(query_length, 1.0, rtol=1e-6, atol=1e-7)
        print(f"\nQuestion {number}: {question}")
        print(f"Question vector shape: {query.shape}; dtype: {query.dtype}")
        print(f"First 12 coordinates: {coordinates(query)}")
        print(f"Vector length: {query_length:.9f}; approximately 1: True")
        displayed.extend((float(value), f"question {number}", j + 1) for j, value in enumerate(query[:12]))
        order, scores, cosines = rank_passages(documents, query, chunks)
        print("\nAll six scores (descending dot product, ascending ID for ties):")
        for i in order:
            print(f"  {chunks[i]['id']} | {chunks[i]['file']} | dot={scores[i]:.9f} | direct cosine={cosines[i]:.9f}")
        print("Dot products and direct cosines agree: True (rtol=1e-6, atol=1e-7)")
        selected = order[:args.k]
        print(f"\nSelected IDs (k={args.k}): {', '.join(chunks[i]['id'] for i in selected)}")
        for i in selected:
            chunk = chunks[i]
            print(f"  [{chunk['id']}] {chunk['file']}: {chunk['text']}")

        if question == Q1:
            best = order[0]
            products = query.astype(np.float64) * documents[best].astype(np.float64)
            first = np.sum(products[:12])
            rest = np.sum(products[12:])
            np.testing.assert_allclose(first + rest, scores[best], rtol=1e-12, atol=1e-12)
            print(f"\nQ1 inspection with top passage {chunks[best]['id']}:")
            for j in range(12):
                print(f"  Coordinate {j + 1:2}: {query[j]:.4f} * {documents[best, j]:.4f} = {products[j]:.9f}")
            print(f"First 12 product subtotal: {first:.9f}")
            print(f"Remaining 372 product subtotal: {rest:.9f}")
            print(f"Full similarity (all 384): {scores[best]:.9f}")
            print(f"Direct cosine from vector lengths: {cosines[best]:.9f}")
            print("Displayed factors are rounded; products use the original unrounded values.")

    smallest = min(displayed, key=lambda item: item[0])
    largest = max(displayed, key=lambda item: item[0])
    print("\nExtremes among the displayed coordinates (documents and questions):")
    for label, (value, owner, coordinate) in (("Smallest", smallest), ("Largest", largest)):
        print(f"  {label}: {value:.4f} ({owner}, coordinate {coordinate})")
    print("Individual coordinates have no assigned semantic names; meaning is represented by the vector as a whole.")
    print("encode_query and encode_document distinguish roles; this model uses the same underlying encoding behavior for both.")
    print("Encoding and retrieval do not generate an answer. Answer generation comes later.")


if __name__ == "__main__":
    main()
