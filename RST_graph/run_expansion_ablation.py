import os
import sys
import json
import asyncio
import logging
import argparse
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.main import RSTRAG

logging.basicConfig(level=logging.WARNING)

CONDITIONS = {
    # A: No expansion at all, no island filtering
    "baseline":       {"use_expansion": False, "probe_weight": 0.5, "top_k_islands": 0},
    # B: HyPE expansion, no island filtering
    "hype":           {"use_expansion": True,  "probe_weight": 0.5, "top_k_islands": 0},
    # C: No expansion, top-5 island filter
    "baseline_island5": {"use_expansion": False, "probe_weight": 0.5, "top_k_islands": 5, "max_bfs_depth": 2},
    # D: HyPE + top-5 island filter (the new hypothesis)
    "hype_island5":   {"use_expansion": True,  "probe_weight": 0.5, "top_k_islands": 5, "max_bfs_depth": 2},
    # E: Baseline 5-Hop (the new challenger)
    "baseline_hop5":  {"use_expansion": False, "probe_weight": 0.5, "top_k_islands": 0, "max_bfs_depth": 5},
    # F: Seed retrieval only, no strong BFS expansion, no weak bridge.
    # Use with --bridging_budget 0 for the "seed retrieval as final evidence" ablation.
    "baseline_seed_only": {"use_expansion": False, "probe_weight": 0.5, "top_k_islands": 0, "max_bfs_depth": 0},
}

def group_questions_by_source(question_list):
    grouped = {}
    for q in question_list:
        source = q.get("source")
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(q)
    return grouped

async def process_corpus_questions(corpus_name, corpus_questions, base_dir, bridging_budget, use_expansion, probe_weight, top_k_islands=0, max_bfs_depth=2):
    workspace_dir = os.path.join(base_dir, corpus_name)
    print(f"\n--- Processing Corpus: {corpus_name} ({len(corpus_questions)} questions) ---")
    
    rag = RSTRAG(workspace_dir)
    rag.retriever.max_bfs_depth = max_bfs_depth
    
    sem = asyncio.Semaphore(15)
    total_len = len(corpus_questions)
    results = []

    async def process_item(idx, item):
        qid = item["id"]
        q = item["question"]
        qtype = item["question_type"]
        gt = item.get("answer", "")

        async with sem:
            answer, context = await rag.query(
                q,
                bridging_budget=bridging_budget,
                use_expansion=use_expansion,
                probe_weight=probe_weight,
                top_k_islands=top_k_islands
            )
            print(f"[{idx + 1}/{total_len}] {qid} ({qtype})")
            return {
                "id": qid, "question": q, "ground_truth": gt,
                "generated_answer": answer, "context": [context],
                "question_type": qtype
            }

    tasks = [process_item(i, item) for i, item in enumerate(corpus_questions)]
    results = await asyncio.gather(*tasks)
    return results

async def run_experiment(args):
    condition = args.condition
    cfg = CONDITIONS[condition]
    use_expansion = cfg["use_expansion"]
    probe_weight  = cfg["probe_weight"]
    top_k_islands = cfg["top_k_islands"]
    max_bfs_depth = cfg.get("max_bfs_depth", 2)

    print(f"\n=== Condition: {condition.upper()} | use_expansion={use_expansion} | probe_weight={probe_weight} | top_k_islands={top_k_islands} | max_bfs_depth={max_bfs_depth} ===")

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if args.source:
        questions = [q for q in questions if q.get("source") == args.source]
    if args.limit:
        questions = questions[:args.limit]

    grouped_questions = group_questions_by_source(questions)
    all_results = []

    for corpus_name, corpus_questions in grouped_questions.items():
        results = await process_corpus_questions(
            corpus_name, corpus_questions, args.base_dir,
            args.bridging_budget, use_expansion, probe_weight, top_k_islands, max_bfs_depth
        )
        all_results.extend(results)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_weak{args.bridging_budget}" if args.bridging_budget != 3 else ""
    out_path = os.path.join(args.output_dir, f"rst_ablation_{condition}{suffix}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Results saved → {out_path}")

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions",  type=str, default=os.path.join(repo_root, "Datasets", "Questions", "novel_questions_test.json"))
    ap.add_argument("--base_dir",   type=str, default=os.path.join(repo_root, "proposition_workspace"))
    ap.add_argument("--output_dir", type=str, default=os.path.join(repo_root, "ablation_results"))
    ap.add_argument("--bridging_budget", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit after source filtering.")
    ap.add_argument("--source", type=str, default=None, help="Optional corpus/source name, e.g. Novel-47676.")
    ap.add_argument(
        "--condition", type=str, default="baseline",
        choices=list(CONDITIONS.keys()),
        help="Experiment condition: " + " | ".join(CONDITIONS.keys())
    )
    args = ap.parse_args()
    asyncio.run(run_experiment(args))
