param(
    [string]$Questions = "Datasets\Questions\novel_questions_test.json",
    [string]$BaseDir = "proposition_workspace",
    [string]$OutputDir = "ablation_results",
    [string]$Source = "",
    [int]$Limit = 0,
    [switch]$RunEval,
    [string]$EvalMode = "API",
    [string]$EvalModel = "gpt-5.4-mini-2026-03-17",
    [string]$EvalBaseUrl = "https://api.openai.com/v1",
    [string]$EvalEmbeddingModel = "text-embedding-3-small"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot

try {
    $commonArgs = @(
        "RST_graph\run_expansion_ablation.py",
        "--questions", $Questions,
        "--base_dir", $BaseDir,
        "--output_dir", $OutputDir,
        "--bridging_budget", "0"
    )

    if ($Source -ne "") {
        $commonArgs += @("--source", $Source)
    }
    if ($Limit -gt 0) {
        $commonArgs += @("--limit", "$Limit")
    }

    conda run -n lightrag python @commonArgs --condition baseline
    conda run -n lightrag python @commonArgs --condition baseline_seed_only

    if ($RunEval) {
        conda run -n bench_eval python Evaluation\generation_eval.py `
            --mode $EvalMode `
            --model $EvalModel `
            --base_url $EvalBaseUrl `
            --embedding_model $EvalEmbeddingModel `
            --data_file (Join-Path $OutputDir "rst_ablation_baseline_weak0.json") `
            --output_file (Join-Path $OutputDir "eval_rst_ablation_baseline_weak0.json")

        conda run -n bench_eval python Evaluation\generation_eval.py `
            --mode $EvalMode `
            --model $EvalModel `
            --base_url $EvalBaseUrl `
            --embedding_model $EvalEmbeddingModel `
            --data_file (Join-Path $OutputDir "rst_ablation_baseline_seed_only_weak0.json") `
            --output_file (Join-Path $OutputDir "eval_rst_ablation_baseline_seed_only_weak0.json")
    }
}
finally {
    Pop-Location
}
