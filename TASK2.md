# Lesson 10 — Task 2: Real embedding retrieval

Run from this project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe embedding_retriever.py
```

The default question is Q1: “May a trained graduate student operate the laser alone at 19:00?” The default selection count is two. To search several questions in one run, supply each as a quoted argument:

```powershell
.\.venv\Scripts\python.exe embedding_retriever.py "First question" "Second question" --k 2
```

The first run downloads the model weights from Hugging Face if needed. The model then runs locally on CPU. There is no external inference API, vector database, or web interface.

## How the code works

The existing `load_chunks()` function reads the six Cedar Lab passages. Importing it does not run the keyword retriever's command-line program. The original keyword file and source files are unchanged.

`SentenceTransformer(..., device="cpu")` loads the real `sentence-transformers/all-MiniLM-L6-v2` model. `encode_document()` processes just the six passage texts. IDs such as A1 and filenames are kept separately so the program can label results without including those labels in the vectors. This document encoding happens once, before the loop over questions, so every question reuses the same document matrix.

The document matrix has shape `(6, 384)`: six rows, one per passage, and 384 numbers per row. `encode_query()` turns each question into a vector with shape `(384,)`. Both calls use `normalize_embeddings=True` and `convert_to_numpy=True`: the results are NumPy arrays scaled to have length approximately one. The script measures these lengths and fails if they are outside the stated tolerance.

Question and passage vectors must have the same number of coordinates because scoring multiplies corresponding positions and adds their products. They must also come from a compatible encoding space: simply having 384 numbers from an unrelated model would not make the positions comparable. Here the same model encodes both roles. `encode_query()` and `encode_document()` distinguish the roles, but this particular model uses the same underlying encoding behavior for both.

For each passage, the dot product multiplies question coordinate 1 by passage coordinate 1, repeats this for all 384 positions, and adds the products. In NumPy, `docs @ query` performs this calculation for all six rows. Since the vectors have length approximately one, these dot products agree with cosine similarity. The program also divides each dot product by the measured question and passage lengths to check cosine directly.

The model produces float32 arrays. Scoring promotes their original values to float64 for arithmetic; this does not invent extra information or round the vectors. Only printed values are rounded. The first 12 coordinates are displayed to four decimal places, while all 384 contribute to retrieval. The Q1 breakdown separately sums the first 12 products and the remaining 372 products and checks that their sum agrees with the full score.

Sorting the six scores from highest to lowest, breaking exact ties by ascending ID, puts the passages in ranked order. Taking the first `k` entries of that order (`order[:args.k]`) turns six scores into two selected passages when `k=2`. There is no relevance threshold: embeddings always return the top `k` available passages, even for an unrelated question. If `k` exceeds six, all six available passages are returned.

Neither the largest nor the smallest displayed coordinate can tell us what a passage means by itself. Coordinates are learned jointly; an individual position has no assigned semantic name. The relationship among all coordinates determines how this model places the text relative to other texts. The displayed 12 values are only a small view of that representation.

Encoding does not turn a document into an answer or answer the question. This program retrieves source passages. Answer generation comes later.

## Evidence

- `evidence/embedding-inspection.txt`: actual standard output of the Q1 run, including versions, shapes, vectors, scores, selected passages, and the arithmetic inspection.
- `evidence/embedding-run-stderr.txt`: separate model-loading diagnostics.
- `evidence/task2-environment.json`: original environment inventory and successful `pip check`; no dependency installation was needed.
- `evidence/task2-original-file-hashes.json`: SHA-256 fingerprints of all original project files, captured before implementation.

## Actual Q1 result

The successful CPU run produced these scores, using all 384 coordinates:

| Rank | Passage | Dot product |
| --- | --- | --- |
| 1 | A2 | 0.686445741 |
| 2 | T1 | 0.581952918 |
| 3 | B1 | 0.532696345 |
| 4 | A1 | 0.374248889 |
| 5 | T2 | 0.270452209 |
| 6 | B2 | 0.128739670 |

With `k=2`, the selected passages are A2 (the two-person rule) and T1 (training renewal). A1 (access hours) ranks fourth. The embedding retriever therefore does not retrieve both A1 and A2 in its top two for this question. The original keyword evidence selected A1 and A2. These actual results show that semantic similarity alone does not guarantee that every needed policy passage appears in the selected context.

For Q1 and A2, the first 12 products sum to `0.016111493`; the remaining 372 sum to `0.670334248`. Together they give the full dot product `0.686445741`. Direct cosine is `0.686445684`; the small difference comes from vector lengths being approximately, rather than exactly, one. All six comparisons passed `rtol=1e-6, atol=1e-7`.

Across the displayed document and question coordinates, the smallest value is `-0.1167` (T1, position 5), and the largest is `0.1357` (T1, position 12). These individual values do not name or explain the meaning of T1. Retrieval compares the complete learned representations.

The environment uses Python 3.14.7, NumPy 2.5.4, sentence-transformers 6.2.0, PyTorch 2.14.1, transformers 5.21.0, and huggingface-hub 2.3.0. The CPU import check also reported PyTorch's local build identifier as `2.14.1+cpu`. All required packages were already installed, and `pip check` found no broken requirements. The initial restricted-network download attempt failed; retrying with network access successfully loaded the real model. No fabricated vectors or scores were used.
