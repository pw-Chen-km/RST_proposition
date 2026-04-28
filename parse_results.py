import json

def load_averages(path, name):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = {}
        for qtype, content in data.items():
            if isinstance(content, dict):
                # LinearRAG has 'average_scores', script output has metrics directly
                avg = content.get("average_scores", content) 
                
                # Exclude detailed lists if any
                if "answer_correctness" in avg or "rouge_score" in avg:
                    results[qtype] = {
                        "Model": name,
                        "Answer Correctness": avg.get("answer_correctness", 0.0),
                        "ROUGE L / Coverage": avg.get("rouge_score", avg.get("coverage_score", 0.0))
                    }
        return results
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def main():
    base_b3_path = "/Volumes/Untitled/GraphRAG-Benchmark/ablation_results/eval_rst_ablation_baseline_REFRESH.json"
    base_b0_path = "/Volumes/Untitled/GraphRAG-Benchmark/ablation_results/eval_rst_ablation_baseline_weak_0_REFRESH.json"
    hype_b3_path = "/Volumes/Untitled/GraphRAG-Benchmark/ablation_results/eval_rst_ablation_hype_REFRESH.json"
    hype_b0_path = "/Volumes/Untitled/GraphRAG-Benchmark/ablation_results/eval_rst_ablation_hype_weak_0_REFRESH.json"

    base_b3_res = load_averages(base_b3_path, "RST-Baseline (Bridge)")
    base_b0_res = load_averages(base_b0_path, "RST-Baseline (No Bridge)")
    hype_b3_res = load_averages(hype_b3_path, "RST-HyPE (Bridge)")
    hype_b0_res = load_averages(hype_b0_path, "RST-HyPE (No Bridge)")

    all_qtypes = set(base_b3_res.keys()) | set(base_b0_res.keys()) | set(hype_b3_res.keys()) | set(hype_b0_res.keys())

    print("### 🏆 RST_graph: HyPE & Weak Bridge 完整 Ablation 公開對決\n")

    for qtype in sorted(all_qtypes):
        col2 = "Coverage" if qtype == "Contextual Summarize" else "ROUGE-L"
        print(f"#### 【{qtype}】")
        print(f"| 架構版本 | Answer Correctness | {col2} |")
        print("|---|---|---|")

        for res in [base_b3_res, base_b0_res, hype_b3_res, hype_b0_res]:
            if qtype in res:
                row = res[qtype]
                print(f"| **{row['Model']}** | {row['Answer Correctness']:.4f} | {row['ROUGE L / Coverage']:.4f} |")
        print("\n")

if __name__ == "__main__":
    main()
