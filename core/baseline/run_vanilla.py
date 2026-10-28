"""
Vanilla baseline: direct LLM prediction without any retrieval.

Reuses the existing labellers with the experience bank disabled, so each
sample is predicted by the plain eval prompts (no KB, no rerank, no RAG).

Usage (from the repo root):
    python core/baseline/run_vanilla.py \
        -t issue \
        -d data/SWE-bench-issue \
        -e <experiment_id> \
        --llm deepseek-chat \
        --embedder qwen3-embedding-8b
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path

# Allow running this script directly: add core/ to sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from datasets import load_dataset

from agent import Agent
from labeller import IssueLabeller
from models import LLM, Embedder
from utils import setup_logger, Config, load_split
from baseline.vanilla_labeller import VanillaTestLabeller

BASELINE_NAME = "vanilla"


def get_output_dir(task: str, experiment_id: str) -> str:
    """
    Get the output directory path for baseline results.

    Args:
        task: Task type ('issue' or 'test')
        experiment_id: Experiment identifier

    Returns:
        Output directory in format 'results/baseline/vanilla/{task}_labeller/{experiment_id}'
    """
    return os.path.join("results", "baseline", BASELINE_NAME, f"{task}_labeller", experiment_id)


def get_labeller(task: str, agent: Agent, dataset, output_path: str):
    """
    Get the appropriate labeller for the task with the experience bank disabled.

    Args:
        task: Task type ('issue' or 'test')
        agent: Agent instance with LLM (no KB)
        dataset: Dataset to process
        output_path: Path to save results (JSONL file)

    Returns:
        IssueLabeller or VanillaTestLabeller instance

    Raises:
        ValueError: If task type is unknown
    """
    ablation_config = {"use_experience_bank": False}
    if task == "issue":
        return IssueLabeller(agent, dataset, output_path, ablation_config=ablation_config)
    elif task == "test":
        return VanillaTestLabeller(agent, dataset, output_path, ablation_config=ablation_config)
    else:
        raise ValueError(f"Unknown task: {task}")


def run_vanilla(
    task: str,
    dataset_path: str,
    experiment_id: str,
    llm_name: str = None,
    embedder_name: str = None,
):
    """
    Run the vanilla baseline on the test dataset.

    Args:
        task: Task type ('issue' or 'test')
        dataset_path: Path to the dataset
        experiment_id: Experiment identifier for organizing results
        llm_name: LLM name from config.yaml
        embedder_name: Embedder name from config.yaml
    """
    config = Config(llm_name=llm_name, embedder_name=embedder_name)

    output_dir = get_output_dir(task, experiment_id)
    log_path = os.path.join(output_dir, "inference.log")
    jsonl_path = os.path.join(output_dir, "inference.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    listener = setup_logger(log_path)

    ds = load_split(dataset_path, split="test")

    llm = LLM(config.llm_model_name, config.llm_base_url)
    # Vanilla baseline uses no knowledge base
    agent = Agent(llm, kb=None)

    labeller = get_labeller(task, agent, list(ds), jsonl_path)

    if len(labeller.dataset) == 0:
        logging.info("All samples already completed. Nothing to do.")
        listener.stop()
        return

    logging.info(f"Mode: Baseline ({BASELINE_NAME}) {'(Resume)' if labeller.is_resume else ''}")
    logging.info(f"Task: {task}")
    logging.info(f"Experiment ID: {experiment_id}")
    logging.info(f"Using LLM Model: {config.llm_model_name}")
    logging.info(f"Dataset: {dataset_path} ({labeller.total_dataset_size} total, {len(labeller.dataset)} remaining)")
    logging.info("Knowledge Base: Disabled (vanilla baseline)")

    asyncio.run(labeller.inference())

    listener.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla baseline: direct LLM prediction without retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-e", "--experiment_id",
        type=str,
        required=True,
        help="Experiment identifier. Results will be saved to results/baseline/vanilla/{task}_labeller/{experiment_id}/"
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
        "--llm",
        type=str,
        required=True,
        help="LLM name from config.yaml (e.g., 'deepseek-chat', 'gpt-4o')"
    )

    parser.add_argument(
        "--embedder",
        type=str,
        required=True,
        help="Embedder name from config.yaml (unused by vanilla but required for consistent configuration)"
    )

    args = parser.parse_args()
    load_dotenv()

    run_vanilla(args.task, args.dataset, args.experiment_id, args.llm, args.embedder)


if __name__ == "__main__":
    main()
