# Cedar Lab RAG — MIT Agentic AI Lesson 10

## PROJECT OVERVIEW

This classroom exercise explores a small retrieval-augmented generation (RAG) workflow. Three source files contain six instructor-provided Cedar Lab passages: A1 and A2 in `docs/access.txt`, T1 and T2 in `docs/training.txt`, and B1 and B2 in `docs/booking.txt`.

1. **Keyword retrieval:** `keyword_retriever.py` counts distinct query words shared with each passage after lowercase tokenization and removal of the prescribed stopwords. It selects up to `k` passages with positive scores.
2. **Embedding retrieval:** `embedding_retriever.py` runs `sentence-transformers/all-MiniLM-L6-v2` locally on CPU. It encodes passage text with `encode_document()` and questions with `encode_query()`, using the same model and normalized NumPy vectors. IDs and filenames remain metadata. The six document vectors are computed once per run and reused across questions.
3. **Comparison:** Both methods rank all six passages by descending score, breaking ties by ascending ID. Embeddings always select the top `k` available passages without a relevance threshold. `results.csv` records the final evaluated Q1–Q4 comparison at `k=2`.
4. **Grounded answer generation and audit:** In separate assistant conversations, the retrieved passages were supplied as the evidence for Q1 and Q4 answers. Saved answers and claim audits distinguish supported statements from missing information. The Python retrievers retrieve passages; they do not generate answers.

The document embedding matrix has shape `(6, 384)` and each question vector has shape `(384,)`. All 384 coordinates contribute to each dot-product score; only the first 12 are displayed. The program checks vector lengths and agreement with directly calculated cosine similarities. Individual coordinates have no assigned semantic names. With this model, `encode_query()` and `encode_document()` distinguish roles but use the same underlying encoding behavior.

The keyword baseline does not match word variants automatically: `student` and `students`, for example, are different tokens. It splits `19:00` into `19` and `00`, so the shared `00` token can contribute to a match with access-hour times.

## ENVIRONMENT AND REPRODUCTION

### Verified environment

The recorded local environment is Windows with an existing `.venv`. Required packages were already installed, and the recorded `pip check` found no broken requirements.

| Component | Verified version |
| --- | --- |
| Python | 3.14.7 |
| sentence-transformers | 6.2.0 |
| NumPy | 2.5.4 |
| PyTorch | 2.14.1 |
| transformers | 5.21.0 |
| huggingface-hub | 2.3.0 |

The PyTorch import check reported the CPU build identifier `2.14.1+cpu`. The model is `sentence-transformers/all-MiniLM-L6-v2`. See `evidence/task2-environment.json` for the full recorded dependency inventory and `TASK2.md` for the encoding and similarity explanation.

### Windows / Git Bash setup

Open Git Bash and enter the project directory. This is the current local path; after extracting the submission elsewhere, replace it with the actual extraction path.

```bash
cd "/c/Users/gueva/OneDrive/Documentos/1_MITCourses/_AgenticAI2026/cedar-rag-task1/cedar-rag"
pwd
source .venv/Scripts/activate
python --version
python -m pip check
```

Reuse the existing `.venv` locally; exclude it from the ZIP. For a freshly extracted copy without an environment, first install Python 3.14.7, then create and activate an environment and install the recorded main package versions:

```bash
py -3.14 -m venv .venv
source .venv/Scripts/activate
python --version
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "sentence-transformers==6.2.0" "numpy==2.5.4" "transformers==5.21.0" "huggingface-hub==2.3.0"
python -m pip check
```

`py -3.14` selects the installed Python 3.14 interpreter; verify the patch version with `python --version`. The commands pin the main packages; the full dependency inventory is recorded separately. These are reproduction instructions, not a claim that packages were reinstalled during documentation updates.

Embedding execution requires the real model weights. The first embedding run may download them from Hugging Face and requires network access. Subsequent runs can use the local model cache. Inference runs locally on CPU, with no external inference API, vector database, or web interface.

### Questions and audit targets

Define these variables in the same Git Bash session used for the commands below:

```bash
Q1="May a trained graduate student operate the laser alone at 19:00?"
Q2="Can I use the laser without another person present?"
Q3="How frequently must I refresh my laser qualification?"
Q4="What is the laser wavelength?"
```

The required IDs are A1 and A2 for Q1, A2 for Q2, and T1 for Q3. Q4's answer is absent from the corpus.

### Keyword and embedding retrieval

Run from the project root in the activated environment. Both scripts support `--k` and default to `k=2`. The keyword script requires one quoted question per invocation. The embedding script accepts multiple quoted questions and defaults to Q1 if none is supplied.

```bash
python keyword_retriever.py "$Q1" --k 2
python embedding_retriever.py "$Q1" --k 2
```

The embedding command also makes the model weights available before the comparison helper runs. It prints versions, vector inspection details, all six scores, selected IDs, and complete selected passages.

### Q1–Q4 comparisons at k=2

To print both methods' results directly:

```bash
for question in "$Q1" "$Q2" "$Q3" "$Q4"; do
    python keyword_retriever.py "$question" --k 2
done
python embedding_retriever.py "$Q1" "$Q2" "$Q3" "$Q4" --k 2
```

The helper uses the same four fixed questions and `k=2`; it takes no question or `--k` arguments:

```bash
python compare_retrieval.py
```

`compare_retrieval.py` forces offline model loading, so run an embedding command with network access first if the weights are not cached. It writes `evidence/retrieval-runs.txt` and `evidence/retrieval-results-data.json`, replacing those files if they already exist. Run it in a separate project copy to preserve submitted evidence. It **does not regenerate `results.csv`** or complete human evaluation fields.

### Q1 k=1 / k=3 experiment

```bash
python embedding_retriever.py "$Q1" --k 1
python embedding_retriever.py "$Q1" --k 3
```

These commands print the two runs. The original captured output and coverage assessment are saved in `evidence/k-experiment.txt`; the commands above do not rewrite that file. Evidence coverage is the number of required IDs retrieved divided by the two required IDs, A1 and A2. This is separate from whether a narrow yes/no conclusion can be inferred from A2 alone.

## RESULTS

Saved local executions produced these selections at `k=2`. IDs and scores appear in rank order. Keyword counts and embedding similarities use different scales.

| Question | Keyword IDs (scores) | Embedding IDs (similarities) | Required evidence coverage |
| --- | --- | --- | --- |
| Q1 | A1 (3), A2 (2) | A2 (0.686445741), T1 (0.581952918) | Keyword complete; embeddings partial, 1/2 |
| Q2 | A2 (2), B1 (1) | A2 (0.585193445), T1 (0.434760706) | Both methods recovered required A2 |
| Q3 | T1 (2), A2 (1) | T1 (0.742027476), B1 (0.471195408) | Both methods recovered required T1 |
| Q4 | A2 (1), B1 (1) | T1 (0.347721381), A2 (0.320827226) | Answer absent from all source documents |

For Q1, embedding coverage remained **50% at k=1, k=2, and k=3**. The selected sets were A2; A2 and T1; and A2, T1, and B1, respectively. A1 ranked fourth. T1 added training-renewal information and B1 added booking rules, but neither supplied the missing access-hours evidence.

The Q1 answer can reject solo operation using A2, but the complete time-and-operation evidence target requires both A1 and A2. The revised answer acknowledges the missing hours schedule within the supplied retrieved context. A1 does contain that schedule in the full corpus.

Q4's wavelength is not specified anywhere in the sources. Embedding retrieval returns nearest passages because it always takes the top `k`, even when no answer is present. A positive score does not establish answerability. The saved Q4 answer states that the wavelength information is missing.

These findings preserve the actual rankings; neither method is assumed to win every comparison.

## EVIDENCE

The `evidence/` folder preserves execution output, environment details, and saved answer/audit records:

| File | Purpose |
| --- | --- |
| `keyword-q1.txt` | Recorded keyword baseline output for Q1 |
| `embedding-inspection.txt` | Real CPU Q1 embedding output: versions, shapes, coordinates, lengths, six scores, and product/cosine inspection |
| `embedding-run-stderr.txt` | Model-loading diagnostics from the embedding inspection run |
| `retrieval-runs.txt` | Recorded keyword and embedding results for Q1–Q4 at k=2 |
| `retrieval-results-data.json` | **Intermediate data** from the original comparison, containing the original blank human evaluation fields |
| `k-experiment.txt` | Actual Q1 outputs at k=1 and k=3, evidence coverage, and passage contributions |
| `answer-q1-original.txt` | Original Q1 answer and claim support notes |
| `answer-q1-revised.txt` | Revised Q1 answer acknowledging missing retrieved access-hours evidence |
| `answer-q4-original.txt` | Q4 answer acknowledging absent wavelength information |
| `answer-audit.txt` | Q1 and Q4 audit language and judgments, subject to student review and confirmation |
| `task2-environment.json` | Recorded environment versions and successful dependency check |
| `task2-original-file-hashes.json` | Historical hashes captured before Task 2 implementation |
| `task2-verification.json` | Historical Task 2 preservation and runtime-check results |

`results.csv` is the **final evaluated comparison**, with four question rows and completed evaluation and observation fields. The intermediate JSON is not the final evaluated artifact. Question wording and targets are also recorded in `retrieval-runs.txt` and `compare_retrieval.py`.

Historical hash evidence describes the files when it was recorded. This README was subsequently updated for submission documentation, so its current hash differs from the original baseline. Historical evidence files have been preserved.

## PARTICIPANTS AND ATTRIBUTION

- **Participant: Guevara Cruz**
- **Working arrangement: Individual academic exercise; no human partner.**
- **AI assistance: ChatGPT and OpenAI Codex.**
- **Student confirmation of final evaluations and answer-audit judgments: Guevara Cruz confirmed the final evaluations in results.csv and answer audits in evidence/answer-audit.txt on October 27, 2026.**

The instructor's “RAG Challenge — Lesson 10” supplied the Cedar Lab challenge, six source passages, task prompts, questions, and audit targets. These instructional materials are credited to the instructor.

The initial keyword retriever was supplied as a **ChatGPT-assisted starter**, then run and validated locally. Subsequent embedding and comparison implementation used **Codex assistance**. ChatGPT also assisted with interpreting the exercise and drafting audit language, subject to student review and confirmation. These contributions are not represented as independently student-authored work.

Recorded embedding vectors, similarity scores, rankings, and selected passages came from **real local CPU model execution**. Assistant-generated implementation guidance, explanations, and draft audit wording are distinct from those execution results. The student must confirm that completed evaluations and language labeled as a human audit accurately reflect their own review before submission. Approval of the individual reflection does not establish confirmation of these separate audit judgments.

## INDIVIDUAL REFLECTION

`reflection.txt` exists in the project root and contains the student's reviewed and approved individual reflection. It contains 144 words by whitespace count, within the maximum of 150 words. The reflection text is preserved unchanged.

## SUBMISSION

Include the following in the final ZIP, preserving the project folder structure:

- `README.md` and `TASK2.md`
- `keyword_retriever.py`, `embedding_retriever.py`, and `compare_retrieval.py`
- `results.csv`, the final evaluated comparison
- `reflection.txt`, the student's approved individual reflection (maximum 150 words)
- `docs/access.txt`, `docs/training.txt`, and `docs/booking.txt`
- The complete `evidence/` folder, including raw retrieval output, the k experiment, Q1/Q4 answers and audits, and supporting JSON and diagnostic files

Exclude `.venv/`, all `__pycache__/` folders, `*.pyc`, and model caches or downloaded model weights. Recreate the environment and obtain model weights using the reproduction instructions above. Confirm any outstanding evaluation and answer-audit judgments and the accuracy of attribution before submission. The individual reflection is approved and included in the submission list. Final ZIP creation and submission remain separate tasks; no ZIP has been created as part of this documentation update.
