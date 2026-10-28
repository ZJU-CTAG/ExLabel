# A Self-Evolving Experience-Guided Assistant for Assessing Task Clarity and Test Coverage in SWE Datasets

An LLM-powered assistant that builds and reuses an experience bank to assess SWE datasets.
It supports two core tasks: **task clarity assessment** and **test coverage assessment** .

## Setup

### 1. Create Environment

```bash
conda create -n ExLabel python=3.12
conda activate ExLabel
pip install -r requirements.txt
```

### 2. Configure Models

Edit `config.yaml` to configure your LLM and embedder models:

```yaml
llms:
  your-llm-name:
    model_name: your-llm-model-identifier
    base_url: your-llm-api-endpoint

embedders:
  your-embedder-name:
    model_name: your-embedder-model-identifier
    base_url: your-embedder-api-endpoint
```

### 3. Set API Keys

Create a `.env` file in the project root:

```env
# LLM API Key
API_KEY=your_llm_api_key

# Embedder API Key
EMBEDDER_API_KEY=your_embedder_api_key
```

## Quick Start

### Reproducing Our Experiments

#### How to run?

Run these Bash commands from the repository root (on Windows, use Git Bash or WSL)
after configuring the model endpoints and
API keys. The examples below use model keys from `config.yaml` and require an
experience bank for each task, built using the accumulation commands below.
Choose the bank and model settings for the experiment you want to reproduce.

```bash
export LLM=deepseek-v4-flash
export EMBEDDER=qwen3-embedding-8b
export KB_ISSUE=kb/exlabel/full_issue_top3.json
export KB_TEST=kb/exlabel/full_test_top3.json
export RERANK_TOPK=3
export RUN_ID="${LLM}_id"
```

The experience banks are not included in the repository. For example, build them
from the provided SWE-bench training splits before running the scripts:

```bash
python core/app.py --task issue --mode accumulation --dataset data/SWE-bench-issue \
  --kb_path "$KB_ISSUE" --experiment_id "${RUN_ID}_build_issue" \
  --llm "$LLM" --embedder "$EMBEDDER" --rerank_topk "$RERANK_TOPK"
python core/app.py --task test --mode accumulation --dataset data/SWE-bench-test \
  --kb_path "$KB_TEST" --experiment_id "${RUN_ID}_build_test" \
  --llm "$LLM" --embedder "$EMBEDDER" --rerank_topk "$RERANK_TOPK"
```

Use separate bank paths for the two tasks. A fresh accumulation run clears the
bank at its output path. Use the same embedder when building and querying a bank.

Use a new `RUN_ID` when changing banks, prompts, or model settings: existing
outputs are resumed by instance ID.

**RQ1: Overall performance.**

```bash
bash scripts/run_effectiveness.sh
```


**RQ2: Component ablation.**

```bash
export RUN_ID="${LLM}_rq2"
bash scripts/run_ablation.sh
```

This uses the configured `LLM` from above. To use a different model, add its key
and endpoint to `config.yaml`, then set `LLM` and a new `RUN_ID`.

**RQ3: Application.**

The 5,000 SWE-rebench samples used in this experiment are available in
[`data/swerebench_5000.jsonl.gz`](data/swerebench_5000.jsonl.gz).
Extract the dataset while keeping the archive:

```bash
gzip -dk data/swerebench_5000.jsonl.gz
```

### Using Your Own Data

If you want to apply ExLabel to your own datasets, follow these steps:

#### 1. Prepare Your Dataset

Place your datasets in the `data/` directory. Each dataset should be in HuggingFace dataset format with `train` and
`test` splits:

- **Accumulation mode** reads the `train` split to build the experience bank
- **Inference mode** reads the `test` split for prediction

**Dataset Schema**

For **Task Clarity Assessment (`issue` task)**:

| Field             | Type   | Description                |
|-------------------|--------|----------------------------|
| instance_id       | string | Unique instance identifier |
| problem_statement | string | Issue description          |
| underspecified    | bool   | Ground truth (true/false)  |
| notes             | list   | list of explanations       |

For **Test Coverage Assessment (`test` task)**:

| Field             | Type   | Description                |
|-------------------|--------|----------------------------|
| instance_id       | string | Unique instance identifier |
| problem_statement | string | Issue description          |
| test_patch        | string | Test patch content         |
| patch             | string | Code patch content         |
| reasonable        | bool   | Ground truth (true/false)  |
| notes             | list   | list of explanations       |

#### 2. Build Your Experience Bank

Before running inference, build an experience bank from your training data:

```bash
# Build knowledge base for issue labelling
python core/app.py \
  --experiment_id your_experiment \
  --task issue \
  --mode accumulation \
  --dataset data/your-dataset \
  --kb_path kb/your_issue_kb.json \
  --llm your-llm-name \
  --embedder your-embedder-name

# Build knowledge base for test labelling
python core/app.py \
  --experiment_id your_experiment \
  --task test \
  --mode accumulation \
  --dataset data/your-dataset \
  --kb_path kb/your_test_kb.json \
  --llm your-llm-name \
  --embedder your-embedder-name
```

#### 3. Run Inference

Use the built experience bank to predict labels on your test set:

```bash
python core/app.py \
  --experiment_id your_experiment \
  --task issue \
  --mode inference \
  --dataset data/your-dataset \
  --kb_path kb/your_issue_kb.json \
  --llm your-llm-name \
  --embedder your-embedder-name
```

For each instance, the system predicts the label and provides a rationale based on retrieved experiences.
