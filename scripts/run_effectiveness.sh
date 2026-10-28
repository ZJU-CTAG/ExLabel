#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Model keys must exist in config.yaml; banks must be built beforehand.
LLM="${LLM:-deepseek-v4-flash}"
EMBEDDER="${EMBEDDER:-qwen3-embedding-8b}"
KB_ISSUE="${KB_ISSUE:-kb/exlabel/full_issue_top3.json}"
KB_TEST="${KB_TEST:-kb/exlabel/full_test_top3.json}"
RERANK_TOPK="${RERANK_TOPK:-3}"
RUN_ID="${RUN_ID:-${LLM}_top${RERANK_TOPK}_v1}"
PYTHON="${PYTHON:-python}"

# Runs the ExLabel component of the overall performance comparison.
for dataset in SWE-bench-issue multiswebench-issue SWE-bench-test multiswebench-test; do
    task="${dataset##*-}"
    kb="$KB_ISSUE"
    if [[ "$task" == test ]]; then kb="$KB_TEST"; fi
    "$PYTHON" core/app.py \
        --task "$task" --mode inference --dataset "data/$dataset" \
        --kb_path "$kb" --experiment_id "${RUN_ID}_effectiveness_${dataset}" \
        --llm "$LLM" --embedder "$EMBEDDER" --rerank_topk "$RERANK_TOPK"
done

echo "Completed: results/inference/{issue,test}_labeller/${RUN_ID}_effectiveness_*/"
