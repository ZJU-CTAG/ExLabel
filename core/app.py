import os
import argparse
import asyncio
import logging
import json
from dotenv import load_dotenv
from datasets import load_dataset

from agent import Agent
from knowledge import KnowledgeBase
from labeller import IssueLabeller, TestLabeller
from models import LLM, Embedder
from utils import setup_logger, Config, load_split

def get_output_dir(task: str, mode: str, experiment_id: str) -> str:
    """
    Get the output directory path for results.

    Args:
        task: Task type
        mode: Mode type
        experiment_id: Experiment identifier

    Returns:
        Output directory path in format 'results/{mode}/{task}_labeller/{experiment_id}'
    """
    return os.path.join("results", mode, f"{task}_labeller", experiment_id)


def get_labeller(task: str, agent: Agent, dataset, output_path: str, ablation_config: dict = None,
                 max_concurrent: int = 100):
    """
    Get the appropriate labeller for the task.

    Args:
        task: Task type ('issue' or 'test')
        agent: Agent instance with LLM and KB
        dataset: Dataset to process
        output_path: Path to save results (JSONL file)
        ablation_config: Optional ablation configuration dictionary
        max_concurrent: Maximum number of concurrent samples

    Returns:
        IssueLabeller or TestLabeller instance

    Raises:
        ValueError: If task type is unknown
    """
    if task not in ("issue", "test"):
        raise ValueError(f"Unknown task: {task}")

    labeller_cls = IssueLabeller if task == "issue" else TestLabeller

    return labeller_cls(agent, dataset, output_path, ablation_config=ablation_config, max_concurrent=max_concurrent)


def run_inference(
    task: str,
    dataset_path: str,
    kb_path: str,
    experiment_id: str,
    llm_name: str = None,
    embedder_name: str = None,
    ablation_config: dict = None,
    rerank_topk: int = 1,
    max_concurrent: int = None,
):
    """
    Run inference on the test dataset using knowledge base.

    Args:
        task: Task type ('issue' or 'test')
        dataset_path: Path to the dataset
        kb_path: Path to the knowledge base
        experiment_id: Experiment identifier for organizing results
        llm_name: LLM name from config.yaml (e.g., 'deepseek-chat', 'gpt-4')
        embedder_name: Embedder name from config.yaml (e.g., 'deepseek-embedder', 'text-embedding-3-small')
        ablation_config: Optional ablation configuration dictionary (for custom inference settings)
        rerank_topk: Number of experiences to keep after reranking (default: 1)
        max_concurrent: Maximum number of concurrent samples (default: 100)
    """
    config = Config(llm_name=llm_name, embedder_name=embedder_name)

    ablation_config = dict(ablation_config) if ablation_config else {}
    ablation_config["rerank_topk"] = rerank_topk

    disable_kb = ablation_config and not ablation_config.get("use_experience_bank", True)

    output_dir = get_output_dir(task, "inference", experiment_id)

    log_path = os.path.join(output_dir, "inference.log")
    jsonl_path = os.path.join(output_dir, "inference.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    listener = setup_logger(log_path)

    ds = load_split(dataset_path, split="test")

    llm = LLM(config.llm_model_name, config.llm_base_url)
    embedder = Embedder(config.embedder_model_name, config.embedder_base_url)

    if disable_kb:
        kb = None
    else:
        kb = KnowledgeBase(kb_path, embedder)
    agent = Agent(llm, kb)

    if max_concurrent is None:
        max_concurrent = 100

    labeller = get_labeller(task, agent, list(ds), jsonl_path, ablation_config,
                            max_concurrent=max_concurrent)

    if len(labeller.dataset) == 0:
        logging.info("All samples already completed. Nothing to do.")
        listener.stop()
        return

    logging.info(f"Mode: Inference {'(Resume)' if labeller.is_resume else ''}")
    logging.info(f"Task: {task}")
    logging.info(f"Experiment ID: {experiment_id}")
    if ablation_config:
        logging.info(f"Custom Config: {json.dumps(ablation_config, indent=2)}")
    logging.info(f"Using LLM Model: {config.llm_model_name}")
    logging.info(f"Using Embedder Model: {config.embedder_model_name}")
    logging.info(f"Dataset: {dataset_path} ({labeller.total_dataset_size} total, {len(labeller.dataset)} remaining)")
    logging.info(f"Knowledge Base: {kb_path if not disable_kb else 'Disabled'}")
    logging.info(f"Rerank Top-K: {rerank_topk}")

    try:
        asyncio.run(labeller.inference())
    finally:
        listener.stop()


def run_ablation(
    task: str,
    dataset_path: str,
    kb_path: str,
    experiment_id: str,
    ablation: str,
    llm_name: str = None,
    embedder_name: str = None,
    rerank_topk: int = 1
):
    """
    Run ablation study on the test dataset.

    Args:
        task: Task type ('issue' or 'test')
        dataset_path: Path to the dataset
        kb_path: Path to the knowledge base
        experiment_id: Experiment identifier for organizing results
        ablation: Ablation setting (disable_kb, disable_rerank, mask_experience, mask_mistakes, mask_both)
        llm_name: LLM name from config.yaml (e.g., 'deepseek-chat', 'gpt-4')
        embedder_name: Embedder name from config.yaml (e.g., 'deepseek-embedder', 'text-embedding-3-small')
        rerank_topk: Number of experiences to keep after reranking (default: 1)
    """
    config = Config(llm_name=llm_name, embedder_name=embedder_name)
    
    ablation_config = None
    disable_kb = False
    
    if ablation == "disable_kb":
        ablation_config = {"use_experience_bank": False, "use_reranking": False, 
                          "mask_experience_field": False, "mask_mistakes_field": False}
        disable_kb = True
    elif ablation == "disable_rerank":
        ablation_config = {"use_experience_bank": True, "use_reranking": False,
                          "mask_experience_field": False, "mask_mistakes_field": False}
    elif ablation == "mask_experience":
        ablation_config = {"use_experience_bank": True, "use_reranking": True,
                          "mask_experience_field": True, "mask_mistakes_field": False}
    elif ablation == "mask_mistakes":
        ablation_config = {"use_experience_bank": True, "use_reranking": True,
                          "mask_experience_field": False, "mask_mistakes_field": True}
    elif ablation == "mask_both":
        ablation_config = {"use_experience_bank": True, "use_reranking": True,
                          "mask_experience_field": True, "mask_mistakes_field": True}
    else:
        raise ValueError(f"Unknown ablation setting: {ablation}. "
                       f"Available: disable_kb, disable_rerank, mask_experience, mask_mistakes, mask_both")

    ablation_config["rerank_topk"] = rerank_topk
    
    output_dir = get_output_dir(task, "ablation", experiment_id)
    log_path = os.path.join(output_dir, "ablation.log")
    jsonl_path = os.path.join(output_dir, "ablation.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    listener = setup_logger(log_path)

    ds = load_split(dataset_path, split="test")

    llm = LLM(config.llm_model_name, config.llm_base_url)
    embedder = Embedder(config.embedder_model_name, config.embedder_base_url)

    if disable_kb:
        kb = None
    else:
        kb = KnowledgeBase(kb_path, embedder)
    agent = Agent(llm, kb)
    
    labeller = get_labeller(task, agent, list(ds), jsonl_path, ablation_config)
    
    if len(labeller.dataset) == 0:
        logging.info("All samples already completed. Nothing to do.")
        listener.stop()
        return

    logging.info(f"Mode: Ablation {'(Resume)' if labeller.is_resume else ''}")
    logging.info(f"Task: {task}")
    logging.info(f"Experiment ID: {experiment_id}")
    logging.info(f"Ablation: {ablation}")
    logging.info(f"Ablation Config: {json.dumps(ablation_config, indent=2)}")
    logging.info(f"Using LLM Model: {config.llm_model_name}")
    logging.info(f"Using Embedder Model: {config.embedder_model_name}")
    logging.info(f"Dataset: {dataset_path} ({labeller.total_dataset_size} total, {len(labeller.dataset)} remaining)")
    logging.info(f"Knowledge Base: {kb_path if not disable_kb else 'Disabled'}")
    logging.info(f"Rerank Top-K: {rerank_topk}")

    asyncio.run(labeller.inference())

    listener.stop()


def run_accumulation(task: str, dataset_path: str, kb_path: str, experiment_id: str, llm_name: str = None, embedder_name: str = None, rerank_topk: int = 1, max_concurrent: int = 100):
    """
    Run accumulation to build the knowledge base from training data.

    Args:
        task: Task type ('issue' or 'test')
        dataset_path: Path to the dataset
        kb_path: Path to the knowledge base
        experiment_id: Experiment identifier for organizing results
        llm_name: LLM name from config.yaml (e.g., 'deepseek-chat', 'gpt-4')
        embedder_name: Embedder name from config.yaml (e.g., 'deepseek-embedder', 'text-embedding-3-small')
        rerank_topk: Number of experiences to keep after reranking (default: 1)
        max_concurrent: Maximum number of concurrent accumulation samples. Lower values
            build the KB more sequentially, letting later samples see earlier experiences
            (recommended for small training sets; default 100)
    """
    config = Config(llm_name=llm_name, embedder_name=embedder_name)

    output_dir = get_output_dir(task, "accumulation", experiment_id)
    log_path = os.path.join(output_dir, "accumulation.log")
    jsonl_path = os.path.join(output_dir, "accumulation.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    listener = setup_logger(log_path)

    ds = load_dataset(dataset_path, split="train")

    llm = LLM(config.llm_model_name, config.llm_base_url)
    embedder = Embedder(config.embedder_model_name, config.embedder_base_url)

    kb = KnowledgeBase(kb_path, embedder)
    agent = Agent(llm, kb)

    ablation_config = {"rerank_topk": rerank_topk} if rerank_topk != 1 else None
    labeller = get_labeller(task, agent, list(ds), jsonl_path, ablation_config,
                            max_concurrent=max_concurrent if max_concurrent else 100)
    if not labeller.is_resume:
        kb.clear()
    
    if len(labeller.dataset) == 0:
        logging.info("All samples already completed. Nothing to do.")
        listener.stop()
        return

    logging.info(f"Mode: Accumulation {'(Resume)' if labeller.is_resume else ''}")
    logging.info(f"Task: {task}")
    logging.info(f"Experiment ID: {experiment_id}")
    logging.info(f"Using LLM Model: {config.llm_model_name}")
    logging.info(f"Using Embedder Model: {config.embedder_model_name}")
    logging.info(f"Dataset: {dataset_path} ({labeller.total_dataset_size} total, {len(labeller.dataset)} remaining)")
    logging.info(f"Knowledge Base: {kb_path}")
    if rerank_topk != 1:
        logging.info(f"Rerank Top-K: {rerank_topk}")

    asyncio.run(labeller.accumulate())

    listener.stop()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-e", "--experiment_id",
        type=str,
        required=True,
        help="Experiment identifier. Results will be saved to results/{mode}/{experiment_id}/"
    )

    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        choices=["issue", "test"],
        help="Task type: 'issue' for issue labelling, 'test' for test labelling"
    )

    parser.add_argument(
        "-m", "--mode",
        type=str,
        required=True,
        choices=["accumulation", "inference", "ablation"],
        help="Mode: 'accumulation' to build KB, 'inference' to predict using KB"
    )
    
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to the dataset"
    )
    
    parser.add_argument(
        "-k", "--kb_path",
        type=str,
        required=True,
        help="Path to the knowledge base"
    )
        
    parser.add_argument(
        "-a", "--ablation",
        type=str,
        default=None,
        choices=["disable_kb", "disable_rerank", "mask_experience", "mask_mistakes", "mask_both"],
        help="Ablation setting: disable_kb, disable_rerank, mask_experience, mask_mistakes, mask_both (can also be used with inference mode)"
    )
    
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="LLM name from config.yaml (e.g., 'deepseek-chat', 'gpt-4')"
    )
    
    parser.add_argument(
        "--embedder",
        type=str,
        required=True,
        help="Embedder name from config.yaml (e.g., 'Qwen3-Embedding-8B')"
    )

    parser.add_argument(
        "--rerank_topk",
        type=int,
        default=1,
        help="Number of experiences to keep after reranking (default: 1)"
    )

    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Maximum number of concurrent samples (default: 100)"
    )

    args = parser.parse_args()
    load_dotenv()

    ablation_config = None
    if args.ablation:
        if args.ablation == "disable_kb":
            ablation_config = {"use_experience_bank": False, "use_reranking": False, 
                              "mask_experience_field": False, "mask_mistakes_field": False}
        elif args.ablation == "disable_rerank":
            ablation_config = {"use_experience_bank": True, "use_reranking": False,
                              "mask_experience_field": False, "mask_mistakes_field": False}
        elif args.ablation == "mask_experience":
            ablation_config = {"use_experience_bank": True, "use_reranking": True,
                              "mask_experience_field": True, "mask_mistakes_field": False}
        elif args.ablation == "mask_mistakes":
            ablation_config = {"use_experience_bank": True, "use_reranking": True,
                              "mask_experience_field": False, "mask_mistakes_field": True}
        elif args.ablation == "mask_both":
            ablation_config = {"use_experience_bank": True, "use_reranking": True,
                              "mask_experience_field": True, "mask_mistakes_field": True}

    if args.mode == "inference":
        run_inference(
            args.task,
            args.dataset,
            args.kb_path,
            args.experiment_id,
            args.llm,
            args.embedder,
            ablation_config,
            args.rerank_topk,
            max_concurrent=args.max_concurrent,
        )
    elif args.mode == "ablation":
        if not args.ablation:
            raise ValueError("--ablation is required when mode is 'ablation'")
        run_ablation(
            args.task,
            args.dataset,
            args.kb_path,
            args.experiment_id,
            args.ablation,
            args.llm,
            args.embedder,
            args.rerank_topk
        )
    elif args.mode == "accumulation":
        run_accumulation(args.task, args.dataset, args.kb_path, args.experiment_id, args.llm, args.embedder, args.rerank_topk, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
