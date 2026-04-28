import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from RST_graph.src.builder import RSTGraphBuilder
from RST_graph.src.main import RSTRAG


DATASETS = [
    ("hotpotqa", "HotpotQA-Smoke"),
    ("2wikimultihopqa", "2WikiMultiHopQA-Smoke"),
    ("musique", "MuSiQue-Smoke"),
]


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def context_from_hotpot_style(item: Dict[str, Any]) -> str:
    sections = []
    for title, sentences in item["context"]:
        text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
        if text:
            sections.append(f"Title: {title}\n{text}")
    return "\n\n".join(sections)


def context_from_musique(item: Dict[str, Any]) -> str:
    sections = []
    for paragraph in item["paragraphs"]:
        title = str(paragraph.get("title", "")).strip()
        text = str(paragraph.get("paragraph_text", "")).strip()
        if text:
            sections.append(f"Title: {title}\n{text}")
    return "\n\n".join(sections)


def first_question(dataset_name: str, data_dir: str) -> Tuple[Dict[str, Any], str]:
    items = read_json(os.path.join(data_dir, f"{dataset_name}.json"))
    item = items[0]
    if dataset_name == "musique":
        return item, context_from_musique(item)
    return item, context_from_hotpot_style(item)


async def run_one(dataset_name: str, source_name: str, data_dir: str, workspace_root: str) -> Dict[str, Any]:
    item, context = first_question(dataset_name, data_dir)
    workspace = os.path.join(workspace_root, source_name)
    os.makedirs(workspace, exist_ok=True)

    builder = RSTGraphBuilder(workspace)
    await builder.ainsert(context)

    rag = RSTRAG(workspace)
    answer, retrieved_context = await rag.query(item["question"], bridging_budget=0)

    return {
        "dataset": dataset_name,
        "workspace": workspace,
        "question": item["question"],
        "ground_truth": item.get("answer", ""),
        "generated_answer": answer,
        "retrieved_context": retrieved_context,
    }


async def main() -> None:
    load_dotenv()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "HippoRAG", "reproduce", "dataset")
    workspace_root = os.path.join(repo_root, "proposition_workspace_smoke")
    output_path = os.path.join(repo_root, "ablation_results", "hipporag_legacy_smoke_one_each.json")

    results: List[Dict[str, Any]] = []
    for dataset_name, source_name in DATASETS:
        print(f"\n=== {dataset_name} ===", flush=True)
        result = await run_one(dataset_name, source_name, data_dir, workspace_root)
        results.append(result)
        print(f"Question: {result['question']}", flush=True)
        print(f"Ground truth: {result['ground_truth']}", flush=True)
        print(f"Generated: {result['generated_answer']}", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved smoke results to {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
