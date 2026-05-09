# RSTGraph HotpotQA / 2Wiki / MuSiQue

This repository is a clean subset for running RST graph experiments on the three HippoRAG legacy benchmark datasets:

- HotpotQA
- 2WikiMultiHopQA
- MuSiQue

The packaged runner builds one RST workspace per dataset source, runs QA, and optionally evaluates generated answers.

## Included

- `RST_graph/`: RST graph building, retrieval, and answer generation.
- `Evaluation/`: generation and retrieval evaluation utilities.
- `Datasets/Corpus/hipporag_legacy_1000.json`: converted corpus file with one source per dataset.
- `Datasets/Questions/hipporag_legacy_1000.json`: converted questions file with 1000 questions per dataset.
- `HippoRAG/reproduce/dataset/`: original HippoRAG legacy source files used by `prepare_hipporag_legacy.py`.

Generated workspaces, caches, vectors, and result JSON files are ignored by git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Optional model overrides:

```text
RST_CHAT_MODEL=gpt-4o-mini
PROPOSITION_EXTRACT_MODEL=gpt-4o-mini
RST_EMBEDDING_MODEL=all-mpnet-base-v2
```

## Quick Smoke Test

Build tiny graphs, run one question per dataset, and run generation evaluation:

```bash
python run_full_pipeline.py --smoke
```

Smoke mode uses:

- 2 passages per source
- 1 question per selected dataset
- 1 evaluation sample per selected dataset

## Full Pipeline

Run all three datasets:

```bash
python run_full_pipeline.py
```

Run selected datasets:

```bash
python run_full_pipeline.py --datasets hotpotqa 2wiki musique
```

Only build and run inference, without evaluation:

```bash
python run_full_pipeline.py --datasets hotpotqa --skip_eval
```

Only run evaluation over existing prediction files:

```bash
python run_full_pipeline.py --datasets hotpotqa --skip_inference
```

Preview commands without running them:

```bash
python run_full_pipeline.py --dry_run
```

## Dataset Routing

The converted benchmark files contain all three datasets, but the runner separates them by `source`:

```text
hotpotqa -> HotpotQA-1000
2wiki    -> 2WikiMultiHopQA-1000
musique  -> MuSiQue-1000
```

Each source gets its own workspace:

```text
rst_workspaces/pipeline/hipporag/HotpotQA-1000-full/
rst_workspaces/pipeline/hipporag/2WikiMultiHopQA-1000-full/
rst_workspaces/pipeline/hipporag/MuSiQue-1000-full/
```

Each workspace contains its own `graph.graphml`, `vector_store.json`, and `weak_index.json`.

## Direct RST Runner

Run one dataset through the lower-level RST dataset runner:

```bash
python run_rst_datasets_experiment.py \
  --source HotpotQA-1000 \
  --max_passages_per_source 0 \
  --output_file results/rst_hotpotqa.json
```

Build graphs only:

```bash
python run_rst_datasets_experiment.py --build_only
```

## Evaluation

Generation evaluation:

```bash
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model all-mpnet-base-v2 \
  --data_file results/rst_hotpotqa.json \
  --output_file results/eval_generation_hotpotqa.json
```

Retrieval evaluation:

```bash
python -m Evaluation.retrieval_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model all-mpnet-base-v2 \
  --data_file results/rst_hotpotqa.json \
  --output_file results/eval_retrieval_hotpotqa.json
```

## Regenerate Converted Dataset Files

If the original files in `HippoRAG/reproduce/dataset/` change, regenerate the converted benchmark files:

```bash
python prepare_hipporag_legacy.py
```

## Notes

- Do not commit `.env`.
- The first graph build calls OpenAI for proposition extraction and uses local `all-mpnet-base-v2` embeddings.
- Re-running without `--force_rebuild` reuses existing workspaces.
