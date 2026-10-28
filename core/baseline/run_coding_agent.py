"""
Coding agent baseline: launch an external coding agent (claude code / codex)
to label each sample with access to the target repository.

Each sample is processed in its own playground worktree (checked out at
`base_commit`); the agent explores the repository read-only and must answer
with a strict JSON label. This baseline makes no LLM API calls from this
codebase — the agent CLI is the labeller.

Usage (from the repo root):
    python core/baseline/run_coding_agent.py \
        -t issue \
        -d data/SWE-bench-issue \
        -e <experiment_id> \
        --agent claude

Preview commands and prompts without launching anything:
    python core/baseline/run_coding_agent.py -t issue -d data/SWE-bench-issue \
        -e <experiment_id> --agent claude --dry-run
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

# Allow running this script directly: add core/ to sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from datasets import load_dataset

from utils import setup_logger
from baseline.rag_utils import ANNOTATION_LABEL
from baseline.coding_agent.cli import build_coding_agent_command
from baseline.coding_agent.labeller import (
    CodingAgentIssueLabeller,
    CodingAgentTestLabeller,
    build_agent_prompt,
)
from baseline.coding_agent.playground import Playground

BASELINE_NAME = "coding_agent"


def _load_dataset(dataset_path: str) -> list:
    """
    Load samples from either a .jsonl file (one JSON object per line) or a
    HuggingFace dataset directory (parquet, with a 'test' split).
    """
    if dataset_path.endswith(".jsonl"):
        samples = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    return list(load_dataset(dataset_path, split="test"))


class _NullLLM:
    """Stand-in for LLM: the coding agent baseline makes no direct API calls."""

    def get_token_stats(self):
        return {
            "total_prompt_tokens": 0,
            "total_cached_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "api_call_count": 0,
        }


class _NullAgent:
    """Stand-in for Agent: only llm.get_token_stats() is read by inference()."""

    llm = _NullLLM()
    kb = None


def get_output_dir(task: str, experiment_id: str) -> str:
    """
    Get the output directory path for baseline results.

    Args:
        task: Task type ('issue' or 'test')
        experiment_id: Experiment identifier

    Returns:
        Output directory in format 'results/baseline/coding_agent/{task}_labeller/{experiment_id}'
    """
    return os.path.join("results", "baseline", BASELINE_NAME, f"{task}_labeller", experiment_id)


def run_dry_run(
    task: str,
    dataset_path: str,
    agent_name: str,
    model: Optional[str],
    output_dir: str,
    limit: Optional[int],
):
    """
    Render prompts and print the commands that would be run, without
    launching any agent, cloning any repo, or consuming any API quota.

    Args:
        task: Task type ('issue' or 'test')
        dataset_path: Path to the dataset
        agent_name: Agent CLI name ('claude' or 'codex')
        model: Optional model override for the agent CLI
        output_dir: Output directory for the rendered prompts
        limit: Only render the first N samples (None for all)
    """
    dataset = _load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]

    prompt_dir = os.path.join(output_dir, "prompts_dryrun")
    os.makedirs(prompt_dir, exist_ok=True)
    label_field = ANNOTATION_LABEL[task]

    logging.info(f"Dry run: {len(dataset)} samples, task={task}, agent={agent_name}")
    for sample in dataset:
        instance_id = sample.get("instance_id", "unknown")
        prompt = build_agent_prompt(task, sample)
        prompt_path = os.path.join(prompt_dir, f"{instance_id}.prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        workdir = os.path.join("playground", "instances", "<experiment_id>", instance_id)
        cmd = build_coding_agent_command(
            agent_name,
            workdir=workdir,
            label_field=label_field,
            model=model,
            out_dir=output_dir,
            instance_id=instance_id,
        )
        printable = subprocess.list2cmdline(cmd) if os.name == "nt" else " ".join(cmd)
        logging.info(f"[{instance_id}] cwd: {workdir}")
        logging.info(f"[{instance_id}] command: {printable}  (prompt via stdin: {prompt_path})")

    logging.info(f"Dry run complete. Rendered prompts saved under: {prompt_dir}")


def run_coding_agent_baseline(
    task: str,
    dataset_path: str,
    experiment_id: str,
    agent_name: str,
    model: Optional[str] = None,
    max_concurrent: int = 8,
    timeout_seconds: float = 900.0,
    retries: int = 2,
    playground_dir: str = "playground",
    keep_worktrees: bool = False,
    limit: Optional[int] = None,
    dry_run: bool = False,
):
    """
    Run the coding agent baseline on the test dataset.

    Args:
        task: Task type ('issue' or 'test')
        dataset_path: Path to the dataset
        experiment_id: Experiment identifier for organizing results
        agent_name: Agent CLI to launch ('claude' or 'codex')
        model: Optional model override for the agent CLI
        max_concurrent: Number of agent instances running in parallel
        timeout_seconds: Per-instance agent timeout in seconds
        retries: Retry attempts per agent session
        playground_dir: Root dir for repo clones and per-instance worktrees
        keep_worktrees: Keep per-instance worktrees after each sample
        limit: Only process the first N samples of the dataset
        dry_run: Render prompts and print commands without launching agents
    """
    output_dir = get_output_dir(task, experiment_id)
    log_path = os.path.join(output_dir, "inference.log")
    jsonl_path = os.path.join(output_dir, "inference.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    listener = setup_logger(log_path)

    try:
        if dry_run:
            run_dry_run(task, dataset_path, agent_name, model, output_dir, limit)
            return

        dataset = _load_dataset(dataset_path)
        if limit is not None:
            dataset = dataset[:limit]

        playground = Playground(playground_dir, experiment_id)
        labeller_cls = {
            "issue": CodingAgentIssueLabeller,
            "test": CodingAgentTestLabeller,
        }[task]
        labeller = labeller_cls(
            _NullAgent(),
            dataset,
            jsonl_path,
            max_concurrent=max_concurrent,
            ablation_config={"retry_attempts": retries},
        )
        labeller.agent_name = agent_name
        labeller.agent_model = model
        labeller.agent_timeout = float(timeout_seconds)
        labeller.playground = playground
        labeller.keep_worktrees = keep_worktrees

        if len(labeller.dataset) == 0:
            logging.info("All samples already completed. Nothing to do.")
            return

        logging.info(f"Mode: Baseline ({BASELINE_NAME}, agent={agent_name}) "
                     f"{'(Resume)' if labeller.is_resume else ''}")
        logging.info(f"Task: {task}")
        logging.info(f"Experiment ID: {experiment_id}")
        logging.info(f"Agent model: {model or 'CLI default'}")
        logging.info(
            f"Dataset: {dataset_path} "
            f"({labeller.total_dataset_size} total, {len(labeller.dataset)} remaining)"
        )
        logging.info(f"Playground: {playground_dir}")
        logging.info(
            f"Concurrency: {max_concurrent}, timeout: {timeout_seconds}s, retries: {retries}"
        )

        try:
            asyncio.run(labeller.inference())
        finally:
            asyncio.run(playground.cleanup())
    finally:
        listener.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Coding agent baseline: label samples with a coding agent over the target repo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-e", "--experiment_id",
        type=str,
        required=True,
        help="Experiment identifier. Results will be saved to "
             "results/baseline/coding_agent/{task}_labeller/{experiment_id}/"
    )

    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        choices=["issue", "test"],
        help="Task type: 'issue' for issue labelling, 'test' for test labelling"
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to the dataset"
    )

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["claude", "codex"],
        help="Coding agent CLI to launch"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model for the agent CLI: an alias (e.g. 'sonnet') or full id (e.g. "
             "'claude-sonnet-5') for claude, a model id for codex "
             "(default: the CLI's own default)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Number of agent instances running in parallel (default: 8)"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-instance agent timeout in seconds (default: 900)"
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry attempts per agent session (default: 2)"
    )

    parser.add_argument(
        "--playground-dir",
        type=str,
        default="playground",
        help="Root dir for repo clones and per-instance worktrees (default: playground)"
    )

    parser.add_argument(
        "--keep-worktrees",
        action="store_true",
        help="Keep per-instance worktrees after each sample (for inspection)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples of the dataset"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render prompts and print commands without launching any agent"
    )

    args = parser.parse_args()
    load_dotenv()

    run_coding_agent_baseline(
        task=args.task,
        dataset_path=args.dataset,
        experiment_id=args.experiment_id,
        agent_name=args.agent,
        model=args.model,
        max_concurrent=args.max_concurrent,
        timeout_seconds=args.timeout,
        retries=args.retries,
        playground_dir=args.playground_dir,
        keep_worktrees=args.keep_worktrees,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
