import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PipelineTask:
    name: str
    corpus_file: str
    questions_file: str
    output_file: str
    eval_output_file: str
    workspace_dir: str
    source: Optional[str] = None


TASKS = [
    PipelineTask(
        name="hotpotqa",
        corpus_file="Datasets/Corpus/hipporag_legacy_1000.json",
        questions_file="Datasets/Questions/hipporag_legacy_1000.json",
        source="HotpotQA-1000",
        output_file="results/rst_hotpotqa.json",
        eval_output_file="results/eval_generation_hotpotqa.json",
        workspace_dir="rst_workspaces/pipeline/hipporag",
    ),
    PipelineTask(
        name="2wiki",
        corpus_file="Datasets/Corpus/hipporag_legacy_1000.json",
        questions_file="Datasets/Questions/hipporag_legacy_1000.json",
        source="2WikiMultiHopQA-1000",
        output_file="results/rst_2wiki.json",
        eval_output_file="results/eval_generation_2wiki.json",
        workspace_dir="rst_workspaces/pipeline/hipporag",
    ),
    PipelineTask(
        name="musique",
        corpus_file="Datasets/Corpus/hipporag_legacy_1000.json",
        questions_file="Datasets/Questions/hipporag_legacy_1000.json",
        source="MuSiQue-1000",
        output_file="results/rst_musique.json",
        eval_output_file="results/eval_generation_musique.json",
        workspace_dir="rst_workspaces/pipeline/hipporag",
    ),
]


def selected_tasks(names: List[str]) -> List[PipelineTask]:
    if not names:
        return TASKS
    wanted = set(names)
    tasks = [task for task in TASKS if task.name in wanted]
    missing = wanted - {task.name for task in tasks}
    if missing:
        raise ValueError(f"Unknown task(s): {', '.join(sorted(missing))}")
    return tasks


def run_command(cmd: List[str], env: dict, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def build_inference_command(task: PipelineTask, args: argparse.Namespace) -> List[str]:
    output_file = task.output_file
    if args.smoke:
        output_file = output_file.replace(".json", "_smoke.json")

    cmd = [
        sys.executable,
        "run_rst_datasets_experiment.py",
        "--corpus_file",
        task.corpus_file,
        "--questions_file",
        task.questions_file,
        "--workspace_dir",
        task.workspace_dir,
        "--output_file",
        output_file,
        "--max_passages_per_source",
        str(args.smoke_passages if args.smoke else args.max_passages_per_source),
        "--bridging_budget",
        str(args.bridging_budget),
        "--max_bfs_depth",
        str(args.max_bfs_depth),
    ]

    if task.source:
        cmd += ["--source", task.source]

    limit = args.smoke_questions if args.smoke else args.limit
    if limit:
        cmd += ["--limit", str(limit)]
    if args.force_rebuild:
        cmd.append("--force_rebuild")
    if args.use_expansion:
        cmd.append("--use_expansion")
    return cmd


def build_eval_command(task: PipelineTask, args: argparse.Namespace) -> List[str]:
    data_file = task.output_file
    eval_output_file = task.eval_output_file
    if args.smoke:
        data_file = data_file.replace(".json", "_smoke.json")
        eval_output_file = eval_output_file.replace(".json", "_smoke.json")

    cmd = [
        sys.executable,
        "-m",
        "Evaluation.generation_eval",
        "--mode",
        "API",
        "--model",
        args.eval_model,
        "--base_url",
        args.base_url,
        "--embedding_model",
        args.eval_embedding_model,
        "--data_file",
        data_file,
        "--output_file",
        eval_output_file,
    ]
    if args.smoke:
        cmd += ["--num_samples", str(args.smoke_eval_samples)]
    elif args.eval_num_samples:
        cmd += ["--num_samples", str(args.eval_num_samples)]
    if args.detailed_eval:
        cmd.append("--detailed_output")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RST graph build, inference, and generation evaluation for benchmark datasets."
    )
    parser.add_argument("--datasets", nargs="*", choices=[task.name for task in TASKS], default=[])
    parser.add_argument("--smoke", action="store_true", help="Run a tiny cheap end-to-end check.")
    parser.add_argument("--smoke_passages", type=int, default=2)
    parser.add_argument("--smoke_questions", type=int, default=1)
    parser.add_argument("--smoke_eval_samples", type=int, default=1)
    parser.add_argument("--max_passages_per_source", type=int, default=0, help="0 means full corpus context.")
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit per selected task.")
    parser.add_argument("--bridging_budget", type=int, default=3)
    parser.add_argument("--max_bfs_depth", type=int, default=2)
    parser.add_argument("--use_expansion", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_model", default="gpt-4o-mini")
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--eval_embedding_model", default="all-mpnet-base-v2")
    parser.add_argument("--eval_num_samples", type=int, default=None)
    parser.add_argument("--detailed_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if "OPENAI_API_KEY" not in env and not args.dry_run:
        raise ValueError("Set OPENAI_API_KEY before running this pipeline.")

    env["RST_EMBEDDING_MODEL"] = "all-mpnet-base-v2"
    # Keep generation_eval on the requested local/HF embedding path.
    env.pop("LLM_API_KEY", None)

    for task in selected_tasks(args.datasets):
        print(f"\n=== {task.name.upper()} ===", flush=True)
        if not args.skip_inference:
            run_command(build_inference_command(task, args), env, args.dry_run)
        if not args.skip_eval:
            run_command(build_eval_command(task, args), env, args.dry_run)

    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
