# RSTGraph HotpotQA / 2Wiki / MuSiQue

This is a clean, GitHub-ready subset of the original `GraphRAG-Benchmark` workspace.
It keeps only the code and data needed to run RST_graph experiments on:

- HotpotQA
- 2WikiMultiHopQA
- MuSiQue

It also keeps the original `Evaluation/` package used for generation and retrieval evaluation.

## What Is Included

- `RST_graph/`: RST graph builder, retrieval, and answer generation.
- `HippoRAG/reproduce/dataset/`: HotpotQA, 2WikiMultiHopQA, and MuSiQue legacy sample JSON files plus corpus files.
- `Datasets/`: prepared benchmark-format files:
  - `Datasets/Corpus/hipporag_legacy_1000.json`
  - `Datasets/Questions/hipporag_legacy_1000.json`
  - HotpotQA distractor sample files
- `Evaluation/`: original evaluation code.

Large raw files over GitHub's 100 MB limit are intentionally excluded.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

PowerShell alternative without creating `.env`:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

Bash/macOS/Linux alternative:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

Do not commit `.env`; it is ignored by `.gitignore`.

## Models

By default, every OpenAI chat/extraction/evaluation call uses `gpt-4o-mini`.
Embeddings use the local HuggingFace model `all-mpnet-base-v2`.

To change the RST answer-generation model, edit `.env`:

```text
RST_CHAT_MODEL=gpt-4o-mini
```

To change the proposition extraction model, edit `.env`:

```text
PROPOSITION_EXTRACT_MODEL=gpt-4o-mini
```

To change the full-pipeline evaluation model, pass:

```bash
python run_full_pipeline.py --eval_model gpt-4o-mini
```

To change the embedding model:

```text
RST_EMBEDDING_MODEL=all-mpnet-base-v2
```

and for evaluation:

```bash
python run_full_pipeline.py --eval_embedding_model all-mpnet-base-v2
```

## Quick Smoke Test

Use the full-pipeline smoke mode to build tiny graphs, run one inference per
dataset, and run generation evaluation:

```bash
python run_full_pipeline.py --smoke
```

## Run RST Experiments

For the `Datasets/Corpus` + `Datasets/Questions` format, each question uses its
`source` field to select the matching `corpus_name` graph:

```bash
python run_rst_datasets_experiment.py --source HotpotQA-1000 --max_passages_per_source 2 --limit 10
```

Without `--source`, the runner builds/reuses one workspace per source and
automatically switches graph per question.

```bash
python run_rst_datasets_experiment.py --max_passages_per_source 2 --limit 30
```

Build graphs only, without running QA:

```bash
python run_rst_datasets_experiment.py --max_passages_per_source 2 --build_only
```

Routing rule:

- `hipporag_legacy_1000`: one pooled graph each for `HotpotQA-1000`, `2WikiMultiHopQA-1000`, and `MuSiQue-1000`.
- `medical`: one graph for `Medical`.
- `novel`: one graph per novel source, e.g. `Novel-30752`, `Novel-44557`, `Novel-47676`.

Examples:

```bash
python run_rst_datasets_experiment.py \
  --corpus_file Datasets/Corpus/medical.json \
  --questions_file Datasets/Questions/medical_questions.json \
  --max_passages_per_source 2 \
  --build_only

python run_rst_datasets_experiment.py \
  --corpus_file Datasets/Corpus/novel.json \
  --questions_file Datasets/Questions/novel_questions.json \
  --source Novel-30752 \
  --max_passages_per_source 2 \
  --build_only
```

Run this no-API local routing test to verify source-to-graph switching:

```bash
python test_datasets_routing.py
```

Run 10 questions over a 200-passage graph for one dataset:

```bash
python run_rst_experiment.py --dataset 2wikimultihopqa --max_passages 200 --num_questions 10
```

Other datasets:

```bash
python run_rst_experiment.py --dataset hotpotqa --max_passages 200 --num_questions 10
python run_rst_experiment.py --dataset musique --max_passages 200 --num_questions 10
```

Run all three:

```bash
python run_rst_experiment.py --dataset all --max_passages 200 --num_questions 10
```

Use `--force_rebuild` if you want to rebuild an existing workspace.

Results are written to `results/` in the format expected by `Evaluation/`.

## Local Builder Test

This uses a short passage and mocked extraction/embeddings, so it does not call OpenAI:

```bash
python test_short_passage_builder.py
```

## Evaluate Results

## Full Pipeline

Run build + inference + generation evaluation for all packaged datasets:

```bash
python run_full_pipeline.py
```

Cheap end-to-end smoke test:

```bash
python run_full_pipeline.py --smoke --force_rebuild
```

Run only selected datasets:

```bash
python run_full_pipeline.py --datasets medical novel --smoke
```

The smoke mode uses 2 passages and 1 question per selected dataset. For novel,
smoke mode checks `Novel-30752`; full mode runs all novel sources.

Generation evaluation:

```bash
python -m Evaluation.generation_eval ^
  --mode API ^
  --model gpt-4o-mini ^
  --base_url https://api.openai.com/v1 ^
  --embedding_model all-mpnet-base-v2 ^
  --data_file results/rst_2wikimultihopqa_200p_10q.json ^
  --output_file results/eval_generation_2wiki.json
```

Retrieval evaluation:

```bash
python -m Evaluation.retrieval_eval ^
  --mode API ^
  --model gpt-4o-mini ^
  --base_url https://api.openai.com/v1 ^
  --embedding_model all-mpnet-base-v2 ^
  --data_file results/rst_2wikimultihopqa_200p_10q.json ^
  --output_file results/eval_retrieval_2wiki.json
```

## Notes

- Full raw HotpotQA/2Wiki dumps were not copied because GitHub rejects files over 100 MB.
- If full datasets are required, use Git LFS or download them separately after cloning.
- The first workspace build calls OpenAI for proposition extraction and uses local `all-mpnet-base-v2` embeddings. Re-running without `--force_rebuild` reuses the saved workspace.
