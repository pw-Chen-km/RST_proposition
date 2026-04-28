import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from prepare_hipporag_legacy import convert_questions
from RST_graph.src.builder import RSTGraphBuilder
from RST_graph.src.main import RSTRAG


DATASET_SOURCES = {
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
    "musique": "MuSiQue",
}


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def passage_text(passage: Dict[str, Any], index: int) -> str:
    title = str(passage.get("title", f"passage-{index}")).strip()
    text = str(passage.get("text") or passage.get("paragraph_text") or "").strip()
    if not text:
        return ""
    return f"Title: {title}\n{text}" if title else text


def format_corpus(passages: List[Dict[str, Any]]) -> str:
    sections = [passage_text(p, i) for i, p in enumerate(passages)]
    return "\n\n".join(section for section in sections if section)


async def build_workspace(workspace: str, context: str, force_rebuild: bool) -> None:
    required = [
        os.path.join(workspace, "graph.graphml"),
        os.path.join(workspace, "vector_store.json"),
        os.path.join(workspace, "weak_index.json"),
    ]
    if not force_rebuild and all(os.path.exists(path) for path in required):
        print(f"Using existing workspace: {workspace}", flush=True)
        return

    os.makedirs(workspace, exist_ok=True)
    builder = RSTGraphBuilder(workspace)
    await builder.ainsert(context)


async def run_one_dataset(args: argparse.Namespace, dataset_name: str) -> str:
    data_dir = os.path.abspath(args.data_dir)
    corpus = read_json(os.path.join(data_dir, f"{dataset_name}_corpus.json"))
    questions_raw = read_json(os.path.join(data_dir, f"{dataset_name}.json"))
    questions_prepared = convert_questions(dataset_name, questions_raw)

    selected_passages = corpus if args.max_passages <= 0 else corpus[: args.max_passages]
    selected_questions = questions_prepared if args.num_questions <= 0 else questions_prepared[: args.num_questions]
    source = f"{DATASET_SOURCES[dataset_name]}-{len(selected_passages)}p"
    workspace = os.path.abspath(os.path.join(args.workspace_dir, source))
    context = format_corpus(selected_passages)

    print(
        f"\n=== {dataset_name} | passages={len(selected_passages)} | "
        f"questions={len(selected_questions)} | workspace={workspace} ===",
        flush=True,
    )

    await build_workspace(workspace, context, args.force_rebuild)

    rag = RSTRAG(workspace)
    rag.retriever.max_bfs_depth = args.max_bfs_depth

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(selected_questions, start=1):
        answer, retrieved_context = await rag.query(
            item["question"],
            bridging_budget=args.bridging_budget,
            use_expansion=args.use_expansion,
            probe_weight=args.probe_weight,
            top_k_islands=args.top_k_islands,
        )
        row = {
            "id": item["id"],
            "source": source,
            "question": item["question"],
            "ground_truth": item.get("answer", ""),
            "generated_answer": answer,
            "context": [retrieved_context],
            "evidence": item.get("evidence", ""),
            "question_type": item.get("question_type", "Complex Reasoning"),
            "dataset": item.get("dataset", DATASET_SOURCES[dataset_name]),
            "original_id": item.get("original_id"),
        }
        results.append(row)
        print(f"[{i}/{len(selected_questions)}] {row['id']} -> {row['generated_answer']}", flush=True)

    output_file = args.output_file
    if not output_file:
        q_label = "allq" if args.num_questions <= 0 else f"{args.num_questions}q"
        p_label = "allp" if args.max_passages <= 0 else f"{args.max_passages}p"
        output_file = os.path.join(args.output_dir, f"rst_{dataset_name}_{p_label}_{q_label}.json")

    if args.dataset == "all":
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_{dataset_name}{ext or '.json'}"

    write_json(output_file, results)
    print(f"Saved: {output_file}", flush=True)
    return output_file


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    datasets = list(DATASET_SOURCES) if args.dataset == "all" else [args.dataset]
    outputs = []
    for dataset_name in datasets:
        outputs.append(await run_one_dataset(args, dataset_name))
    print("\nFinished:")
    for path in outputs:
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RST_graph workspaces and run QA on HotpotQA/2Wiki/MuSiQue.")
    parser.add_argument("--dataset", choices=["all", *sorted(DATASET_SOURCES)], default="2wikimultihopqa")
    parser.add_argument("--data_dir", default=os.path.join("HippoRAG", "reproduce", "dataset"))
    parser.add_argument("--workspace_dir", default="rst_workspaces")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_passages", type=int, default=200, help="Use 0 or a negative value for all passages.")
    parser.add_argument("--num_questions", type=int, default=10, help="Use 0 or a negative value for all questions.")
    parser.add_argument("--bridging_budget", type=int, default=3)
    parser.add_argument("--max_bfs_depth", type=int, default=2)
    parser.add_argument("--top_k_islands", type=int, default=0)
    parser.add_argument("--probe_weight", type=float, default=0.5)
    parser.add_argument("--use_expansion", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
