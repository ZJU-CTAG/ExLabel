"""
SimpleRAG baseline: retrieval-augmented prediction over the training set.

The raw task input is used directly as the retrieval key. The most similar
labelled training samples are retrieved with a single retrieval mode
(--retrieval dense|bm25, no score fusion) and their annotations
(label + notes) are injected as few-shot examples for prediction.

The index over the train split is built automatically on first run and
cached under kb/simplerag/<task>.json; use --rebuild to force rebuilding.

Usage (from the repo root):
    python core/baseline/run_simple_rag.py \
        -t issue \
        -d data/SWE-bench-issue \
        -e <experiment_id> \
        --llm deepseek-chat \
        --embedder qwen3-embedding-8b \
        --retrieval dense
"""

import os
import sys
import logging
import argparse
import asyncio
from pathlib import Path

# Allow running this script directly: add core/ to sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from datasets import load_dataset

from agent import Agent
from knowledge import KnowledgeBase
from models import LLM, Embedder
from utils import setup_logger, Config, load_split

from baseline.rag_utils import (
    build_input_key,
    build_annotation_value,
    is_labelled,
)
from baseline.simple_rag.retrievers import get_retriever

BASELINE_NAME = "simple_rag"
DEFAULT_INDEX_DIR = "kb/simplerag"


def get_output_dir(task: str, experiment_id: str) -> str:
    """
    Get the output directory path for SimpleRAG results.

    Args:
        task: Task type ('issue' or 'test')
        experiment_id: Experiment identifier

    Returns:
        Output directory in format 'results/baseline/simple_rag/{task}_labeller/{experiment_id}'
    """
    return os.path.join("results", "baseline", BASELINE_NAME, f"{task}_labeller", experiment_id)


def get_labeller(task: str, agent: Agent, dataset, output_path: str):
    """
    Get the appropriate SimpleRAG labeller for the task.

    Args:
        task: Task type ('issue' or 'test')
        agent: Agent instance with LLM and the SimpleRAG index as KB
        dataset: Dataset to process
        output_path: Path to save results (JSONL file)

    Returns:
        SimpleRAGIssueLabeller or SimpleRAGTestLabeller instance

    Raises:
        ValueError: If task type is unknown
    """
    from baseline.simple_rag.labeller import SimpleRAGIssueLabeller, SimpleRAGTestLabeller

    if task == "issue":
        return SimpleRAGIssueLabeller(agent, dataset, output_path)
    elif task == "test":
        return SimpleRAGTestLabeller(agent, dataset, output_path)
    else:
        raise ValueError(f"Unknown task: {task}")


async def ensure_index(
    index_path: str,
    dataset_path: str,
    task: str,
    embedder,
    rebuild: bool = False,
) -> KnowledgeBase:
    """
    Load the SimpleRAG index, building it from the train split if needed.

    Args:
        index_path: Path to the cached index JSON file
        dataset_path: Path to the dataset whose train split populates the index
        task: Task type ('issue' or 'test')
        embedder: Embedder instance used to compute embeddings
        rebuild: Force rebuilding the index even if a cache exists

    Returns:
        KnowledgeBase ready for retrieval
    """
    kb = KnowledgeBase(index_path, embedder)

    if rebuild:
        logging.info(f"Rebuilding SimpleRAG index at {index_path}")
        kb.clear()

    if kb.memories:
        logging.info(
            f"Loaded SimpleRAG index from {index_path} ({len(kb.memories)} labelled samples)"
        )
        return kb

    if dataset_path.lower().endswith(".jsonl"):
        raise ValueError(
            "A standalone JSONL file has no labelled train split. Provide an "
            "existing SimpleRAG index with --index."
        )
    ds = load_split(dataset_path, split="train")
    train_samples = [s for s in list(ds) if is_labelled(s, task)]
    skipped = len(ds) - len(train_samples) if hasattr(ds, "__len__") else 0

    logging.info(
        f"Building SimpleRAG index from {dataset_path} train split "
        f"({len(train_samples)} labelled samples{f', {skipped} unlabelled skipped' if skipped else ''})"
    )

    for i, sample in enumerate(train_samples, 1):
        key = build_input_key(sample, task)
        value = build_annotation_value(sample, task)
        success, _ = await kb.add_memory(key, value)
        if not success:
            logging.warning(f"Failed to add sample {sample.get('instance_id', i)} to the index")
        if i % 50 == 0 or i == len(train_samples):
            logging.info(f"Indexed {i}/{len(train_samples)} samples")

    kb.save()
    return kb


def run_simple_rag(
    task: str,
    dataset_path: str,
    experiment_id: str,
    llm_name: str = None,
    embedder_name: str = None,
    index_path: str = None,
    rag_topk: int = 3,
    rebuild: bool = False,
    retrieval_mode: str = "dense",
):
    """
    Run the SimpleRAG baseline on the test dataset.

    Args:
        task: Task type ('issue' or 'test')
        dataset_path: Path to the dataset
        experiment_id: Experiment identifier for organizing results
        llm_name: LLM name from config.yaml
        embedder_name: Embedder name from config.yaml
        index_path: Path to the cached SimpleRAG index (default: kb/simplerag/<task>.json)
        rag_topk: Number of annotated training samples injected per prediction
        rebuild: Force rebuilding the index from the train split
        retrieval_mode: Single retrieval mode, 'dense' or 'bm25' (no fusion)
    """

    async def _main():
        nonlocal index_path
        config = Config(llm_name=llm_name, embedder_name=embedder_name)

        output_dir = get_output_dir(task, experiment_id)
        log_path = os.path.join(output_dir, "inference.log")
        jsonl_path = os.path.join(output_dir, "inference.jsonl")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        listener = setup_logger(log_path)

        try:
            if index_path is None:
                index_path = os.path.join(DEFAULT_INDEX_DIR, f"{task}.json")

            embedder = Embedder(config.embedder_model_name, config.embedder_base_url)
            kb = await ensure_index(index_path, dataset_path, task, embedder, rebuild)

            llm = LLM(config.llm_model_name, config.llm_base_url)
            agent = Agent(llm, kb=kb)

            ds = load_split(dataset_path, split="test")
            labeller = get_labeller(task, agent, list(ds), jsonl_path)
            labeller.rag_topk = max(1, int(rag_topk))
            labeller.retriever = get_retriever(retrieval_mode, kb, embedder)

            if len(labeller.dataset) == 0:
                logging.info("All samples already completed. Nothing to do.")
                return

            logging.info(f"Mode: Baseline ({BASELINE_NAME}) {'(Resume)' if labeller.is_resume else ''}")
            logging.info(f"Task: {task}")
            logging.info(f"Experiment ID: {experiment_id}")
            logging.info(f"Using LLM Model: {config.llm_model_name}")
            logging.info(f"Using Embedder Model: {config.embedder_model_name}")
            logging.info(f"Dataset: {dataset_path} ({labeller.total_dataset_size} total, {len(labeller.dataset)} remaining)")
            logging.info(f"SimpleRAG Index: {index_path} ({len(kb.memories)} labelled samples)")
            logging.info(f"Retrieval Mode: {retrieval_mode}")
            logging.info(f"Retrieval Top-K: {labeller.rag_topk}")

            await labeller.inference()
        finally:
            listener.stop()

    asyncio.run(_main())


def main():
    parser = argparse.ArgumentParser(
        description="SimpleRAG baseline: retrieve similar labelled training samples and predict with their annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-e", "--experiment_id",
        type=str,
        required=True,
        help="Experiment identifier. Results will be saved to results/baseline/simple_rag/{task}_labeller/{experiment_id}/"
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
        "-k", "--index",
        type=str,
        default=None,
        help="Path to the SimpleRAG index file (default: kb/simplerag/<task>.json)"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of annotated training samples retrieved per prediction (default: 3)"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the index from the train split"
    )

    parser.add_argument(
        "--retrieval",
        type=str,
        default="dense",
        choices=["dense", "bm25"],
        help="Single retrieval mode: dense (vector cosine) or bm25 (lexical); no fusion (default: dense)"
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
        help="Embedder name from config.yaml (e.g., 'qwen3-embedding-8b')"
    )

    args = parser.parse_args()
    load_dotenv()

    run_simple_rag(
        args.task,
        args.dataset,
        args.experiment_id,
        args.llm,
        args.embedder,
        index_path=args.index,
        rag_topk=args.topk,
        rebuild=args.rebuild,
        retrieval_mode=args.retrieval,
    )


if __name__ == "__main__":
    main()
