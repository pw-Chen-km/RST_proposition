import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from RST_graph.src.builder import RSTGraphBuilder
from RST_graph.src.main import RSTRAG


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_workspace_name(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_") or "unnamed_source"


def group_questions_by_source(questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in questions:
        source = item.get("source")
        if not source:
            raise ValueError(f"Question is missing source: {item.get('id', item)}")
        grouped.setdefault(str(source), []).append(item)
    return grouped


def corpus_by_source(corpus_items: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(corpus_items, dict):
        corpus_items = [corpus_items]

    mapped = {}
    for item in corpus_items:
        name = item.get("corpus_name") or item.get("source")
        if not name:
            raise ValueError(f"Corpus item is missing corpus_name/source: {item}")
        mapped[str(name)] = item
    return mapped


def limit_context_passages(context: str, max_passages: int) -> str:
    if max_passages <= 0:
        return context

    blocks = re.split(r"\n\s*\n(?=Title:\s*)", context.strip())
    selected = [block.strip() for block in blocks if block.strip()]
    if len(selected) > 1:
        return "\n\n".join(selected[:max_passages])

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", context.strip()) if block.strip()]
    if len(paragraphs) > 1:
        return "\n\n".join(paragraphs[:max_passages])

    approx_chars_per_passage = 1200
    return context[: max_passages * approx_chars_per_passage]


def workspace_ready(workspace: str) -> bool:
    required = ["graph.graphml", "vector_store.json", "weak_index.json"]
    return all(os.path.exists(os.path.join(workspace, name)) for name in required)


async def build_workspace_for_source(
    source: str,
    corpus_item: Dict[str, Any],
    workspace_dir: str,
    force_rebuild: bool,
    max_passages_per_source: int,
) -> str:
    suffix = f"{max_passages_per_source}p" if max_passages_per_source > 0 else "full"
    workspace = os.path.abspath(os.path.join(workspace_dir, f"{safe_workspace_name(source)}-{suffix}"))
    if not force_rebuild and workspace_ready(workspace):
        print(f"Using existing workspace for {source}: {workspace}", flush=True)
        return workspace

    context = str(corpus_item.get("context") or corpus_item.get("text") or "")
    context = limit_context_passages(context, max_passages_per_source)
    if not context.strip():
        raise ValueError(f"Corpus source has empty context: {source}")

    print(
        f"Building workspace for {source}: {workspace} "
        f"(passage_limit={max_passages_per_source or 'full'})",
        flush=True,
    )
    os.makedirs(workspace, exist_ok=True)
    builder = RSTGraphBuilder(workspace)
    await builder.ainsert(context)
    return workspace


async def run_source_questions(
    source: str,
    questions: List[Dict[str, Any]],
    workspace: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rag = RSTRAG(workspace)
    rag.retriever.max_bfs_depth = args.max_bfs_depth

    results = []
    for i, item in enumerate(questions, start=1):
        answer, retrieved_context = await rag.query(
            item["question"],
            bridging_budget=args.bridging_budget,
            use_expansion=args.use_expansion,
            probe_weight=args.probe_weight,
            top_k_islands=args.top_k_islands,
        )
        row = {
            "id": item.get("id", f"{source}-{i}"),
            "source": source,
            "question": item["question"],
            "ground_truth": item.get("answer", item.get("ground_truth", "")),
            "generated_answer": answer,
            "context": [retrieved_context],
            "evidence": item.get("evidence", ""),
            "question_type": item.get("question_type", "Complex Reasoning"),
            "dataset": item.get("dataset"),
            "original_id": item.get("original_id"),
        }
        results.append(row)
        print(f"[{source} {i}/{len(questions)}] {row['id']} -> {answer}", flush=True)
    return results


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    corpus_items = read_json(args.corpus_file)
    corpus_lookup = corpus_by_source(corpus_items)

    if args.build_only:
        sources = [args.source] if args.source else list(corpus_lookup)
        for source in sources:
            if source not in corpus_lookup:
                raise KeyError(f"No corpus item found for source: {source}")
            await build_workspace_for_source(
                source,
                corpus_lookup[source],
                args.workspace_dir,
                args.force_rebuild,
                args.max_passages_per_source,
            )
        print("\nBuild-only run finished.", flush=True)
        return

    questions = read_json(args.questions_file)

    if args.source:
        questions = [q for q in questions if q.get("source") == args.source]
    if args.limit:
        questions = questions[: args.limit]

    grouped_questions = group_questions_by_source(questions)

    all_results: List[Dict[str, Any]] = []
    for source, source_questions in grouped_questions.items():
        if source not in corpus_lookup:
            raise KeyError(f"No corpus item found for question source: {source}")
        workspace = await build_workspace_for_source(
            source,
            corpus_lookup[source],
            args.workspace_dir,
            args.force_rebuild,
            args.max_passages_per_source,
        )
        all_results.extend(await run_source_questions(source, source_questions, workspace, args))

    output_file = args.output_file or os.path.join(args.output_dir, "rst_datasets_results.json")
    write_json(output_file, all_results)
    print(f"\nSaved: {output_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RST_graph on GraphRAG-Benchmark Datasets/Corpus + Datasets/Questions files."
    )
    parser.add_argument("--corpus_file", default=os.path.join("Datasets", "Corpus", "hipporag_legacy_1000.json"))
    parser.add_argument("--questions_file", default=os.path.join("Datasets", "Questions", "hipporag_legacy_1000.json"))
    parser.add_argument("--workspace_dir", default=os.path.join("rst_workspaces", "datasets"))
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--source", default=None, help="Optional source/corpus_name filter, e.g. HotpotQA-1000.")
    parser.add_argument("--limit", type=int, default=None, help="Optional global question limit after source filtering.")
    parser.add_argument("--max_passages_per_source", type=int, default=0, help="Use only the first N passages per source. 0 means full source context.")
    parser.add_argument("--bridging_budget", type=int, default=3)
    parser.add_argument("--max_bfs_depth", type=int, default=2)
    parser.add_argument("--top_k_islands", type=int, default=0)
    parser.add_argument("--probe_weight", type=float, default=0.5)
    parser.add_argument("--use_expansion", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--build_only", action="store_true", help="Only build/reuse source workspaces; do not run QA.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
