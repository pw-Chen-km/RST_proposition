import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from RST_graph.src.builder import RSTGraphBuilder
from RST_graph.src.main import RSTRAG


DATASET_SOURCES = {
    "hotpotqa": "HotpotQA-Diagnostic",
    "2wikimultihopqa": "2WikiMultiHopQA-Diagnostic",
    "musique": "MuSiQue-Diagnostic",
}


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_corpus(passages: List[Dict[str, Any]]) -> str:
    sections = []
    for i, passage in enumerate(passages):
        title = str(passage.get("title", f"passage-{i}")).strip()
        text = str(passage.get("text", "")).strip()
        if text:
            sections.append(f"Title: {title}\n{text}")
    return "\n\n".join(sections)


async def build_workspace(workspace: str, context: str, force_rebuild: bool) -> None:
    graph_path = os.path.join(workspace, "graph.graphml")
    vector_path = os.path.join(workspace, "vector_store.json")
    weak_path = os.path.join(workspace, "weak_index.json")
    if not force_rebuild and all(os.path.exists(path) for path in [graph_path, vector_path, weak_path]):
        print(f"Using existing workspace: {workspace}", flush=True)
        return

    os.makedirs(workspace, exist_ok=True)
    builder = RSTGraphBuilder(workspace)
    await builder.ainsert(context)


async def run_diagnostic(args: argparse.Namespace) -> None:
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

    data_dir = os.path.abspath(args.data_dir)
    source = args.source_name or f"{DATASET_SOURCES[args.dataset]}-{args.max_passages}"
    workspace = os.path.abspath(os.path.join(args.base_dir, source))

    corpus = read_json(os.path.join(data_dir, f"{args.dataset}_corpus.json"))
    questions = read_json(os.path.join(data_dir, f"{args.dataset}.json"))
    selected_passages = corpus[: args.max_passages]
    selected_questions = questions[: args.num_questions]
    context = format_corpus(selected_passages)

    print(
        f"Dataset={args.dataset} | passages={len(selected_passages)} | "
        f"questions={len(selected_questions)} | chars={len(context)} | source={source}",
        flush=True,
    )

    await build_workspace(workspace, context, args.force_rebuild)

    rag = RSTRAG(workspace)
    rag.retriever.max_bfs_depth = args.max_bfs_depth

    results = []
    for i, item in enumerate(selected_questions, start=1):
        answer, retrieved_context = await rag.query(
            item["question"],
            bridging_budget=args.bridging_budget,
            use_expansion=args.use_expansion,
            top_k_islands=args.top_k_islands,
        )
        row = {
            "id": str(item.get("_id") or item.get("id") or i),
            "source": source,
            "question": item["question"],
            "ground_truth": item.get("answer", ""),
            "generated_answer": answer,
            "context": [retrieved_context],
            "question_type": "Complex Reasoning",
        }
        results.append(row)
        print(f"\n[{i}/{len(selected_questions)}] {row['id']}", flush=True)
        print(f"Q: {row['question']}", flush=True)
        print(f"GT: {row['ground_truth']}", flush=True)
        print(f"A: {row['generated_answer']}", flush=True)
        print(f"context chars: {len(retrieved_context)}", flush=True)

    out_path = args.output_file or os.path.join(
        "ablation_results",
        f"hipporag_pooled_diagnostic_{args.dataset}_{args.max_passages}p_{args.num_questions}q.json",
    )
    write_json(out_path, results)
    print(f"\nSaved diagnostic results to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small pooled HippoRAG corpus graph and run RST inference.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SOURCES), default="2wikimultihopqa")
    parser.add_argument("--data_dir", default=os.path.join("HippoRAG", "reproduce", "dataset"))
    parser.add_argument("--base_dir", default="proposition_workspace_diagnostic")
    parser.add_argument("--source_name", default=None)
    parser.add_argument("--max_passages", type=int, default=200)
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--bridging_budget", type=int, default=3)
    parser.add_argument("--max_bfs_depth", type=int, default=2)
    parser.add_argument("--top_k_islands", type=int, default=0)
    parser.add_argument("--use_expansion", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()
    asyncio.run(run_diagnostic(args))


if __name__ == "__main__":
    main()
