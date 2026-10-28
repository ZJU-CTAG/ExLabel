import asyncio
import logging
import os
import json
import threading
from inspect import isawaitable
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from utils.json_parser import parse_json_output


class BaseLabeller(ABC):
    """
    Abstract base class for labellers (issue labelling, test labelling, etc.).
    
    Provides core infrastructure for dataset processing with:
    - Concurrent task execution with configurable limits
    - Batch processing with progress tracking
    - LLM interaction with automatic response parsing
    - Knowledge base integration (optional)
    - Result collection and persistence
    
    Subclasses must implement task-specific logic by overriding abstract methods.
    """
    
    def __init__(
        self, 
        agent, 
        dataset: List[Dict[str, Any]], 
        output_path: Optional[str] = None,
        max_concurrent: int = 100, 
        ablation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the labeller with agent, dataset, and execution parameters.
        
        Args:
            agent: Agent instance providing LLM and optional knowledge base access
            dataset: List of samples to process (will be filtered if resuming)
            output_path: Path to save results as JSONL file (None to skip saving)
            max_concurrent: Maximum number of concurrent tasks (default: 100)
            ablation_config: Configuration for ablation study variants:
                - use_experience_bank: Whether to use experience bank (default: True)
                - use_reranking: Whether to use re-ranking (default: True)
                - mask_experience_field: Whether to mask experience field (default: False)
                - mask_mistakes_field: Whether to mask mistakes field (default: False)
                - rerank_topk: Number of experiences kept after reranking (default: 1)
        """
        self.agent = agent
        self.max_concurrent = max_concurrent
        
        self.ablation_config = ablation_config or {}
        self.use_experience_bank = self.ablation_config.get('use_experience_bank', True)
        self.use_reranking = self.ablation_config.get('use_reranking', True)
        self.mask_experience_field = self.ablation_config.get('mask_experience_field', False)
        self.mask_mistakes_field = self.ablation_config.get('mask_mistakes_field', False)
        self.retry_attempts = max(1, int(self.ablation_config.get('retry_attempts', 3)))
        self.retry_delay_seconds = float(self.ablation_config.get('retry_delay_seconds', 0.5))
        self.rerank_topk = max(1, int(self.ablation_config.get('rerank_topk', 1)))

        self._concurrency_limiter = asyncio.Semaphore(max_concurrent)
        
        self.success_count = 0
        self.failure_count = 0
        self.results = []
        self.current_output_path = output_path
        self.debug_prompts_dir = None
        self._write_lock = threading.Lock()
        self.operation_dict = {"insert": 0, "modify": 0, "delete": 0, "unknown": 0}
        
        self.is_resume = False
        self.completed_ids = set()
        self.total_dataset_size = len(dataset)
        
        if self.current_output_path is not None:
            output_dir = os.path.dirname(self.current_output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            if os.path.exists(self.current_output_path):
                self.is_resume = True
                self.completed_ids = self._load_completed_instance_ids()
                
            if not self.is_resume:
                with open(self.current_output_path, 'w', encoding='utf-8') as f:
                    pass
            logging.info(f"Writing results to: {self.current_output_path} (append={self.is_resume})")
            
            self.debug_prompts_dir = os.path.join(output_dir, f"prompts_debug")
            os.makedirs(self.debug_prompts_dir, exist_ok=True)
            logging.info(f"Debug prompts will be saved to: {self.debug_prompts_dir}")
        else:
            self.debug_prompts_dir = None
        
        self.dataset = self._filter_dataset(dataset)
        
        self.debug_lock = asyncio.Lock()

    @staticmethod
    def _rerank_candidate_id(index: int) -> str:
        """Return a compact, spreadsheet-style identifier for a rerank candidate."""
        if index < 0:
            raise ValueError("Candidate index must be non-negative")

        # A, B, ..., Z, AA, AB, ... -- short labels are less error-prone for
        # an LLM to reproduce than the underlying (often commit-SHA) IDs.
        value = index + 1
        label = ""
        while value:
            value, remainder = divmod(value - 1, 26)
            label = chr(ord("A") + remainder) + label
        return label

    def _prepare_rerank_candidates(
        self, kb_entries: List[Dict[str, Any]]
    ) -> tuple[str, Dict[str, Dict[str, Any]]]:
        """Serialize rerank candidates with LLM-facing aliases, not KB IDs.

        The returned mapping keeps the original entry IDs internal so a model's
        selected alias can be resolved after its response is parsed.
        """
        candidate_entries = {}
        serialized_candidates = []

        for index, entry in enumerate(kb_entries):
            candidate_id = self._rerank_candidate_id(index)
            candidate_entries[candidate_id] = entry
            candidate = {
                "candidate_id": candidate_id,
                "perspective": entry.get("perspective", ""),
            }
            if not self.mask_experience_field:
                candidate["experience"] = entry.get("experience", "")
            if "mistakes" in entry and not self.mask_mistakes_field:
                candidate["mistakes"] = entry["mistakes"]
            serialized_candidates.append(json.dumps(candidate, ensure_ascii=False))

        return "\n".join(serialized_candidates), candidate_entries

    async def _retry_step(
        self,
        step_name: str,
        step_func,
        instance_id: str = "unknown",
        *,
        retry_on_none: bool = True,
        retry_on_empty: bool = False,
        attempts: Optional[int] = None,
        delay_seconds: Optional[float] = None,
    ):
        """
        Execute a step with retry support.

        Args:
            step_name: Human-readable step name for logging
            step_func: Callable returning result or awaitable result
            instance_id: Sample identifier for logs
            retry_on_none: Whether to retry when result is None
            retry_on_empty: Whether to retry when result is an empty list/dict/string
            attempts: Override default retry attempts
            delay_seconds: Override default delay between retries

        Returns:
            Step result if successful, otherwise the final failed result (typically None)
        """
        max_attempts = max(1, attempts if attempts is not None else self.retry_attempts)
        retry_delay = self.retry_delay_seconds if delay_seconds is None else max(0.0, delay_seconds)

        def _needs_retry(result: Any) -> bool:
            if retry_on_none and result is None:
                return True
            if retry_on_empty and isinstance(result, (list, dict, str)) and len(result) == 0:
                return True
            return False

        last_result = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = step_func()
                if isawaitable(result):
                    result = await result

                if not _needs_retry(result):
                    if attempt > 1:
                        logging.info(
                            f"[{instance_id}] Step '{step_name}' succeeded on retry {attempt}/{max_attempts}"
                        )
                    return result

                last_result = result
                if attempt < max_attempts:
                    logging.warning(
                        f"[{instance_id}] Step '{step_name}' returned empty result on attempt "
                        f"{attempt}/{max_attempts}, retrying..."
                    )
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)
            except Exception as e:
                if attempt < max_attempts:
                    logging.warning(
                        f"[{instance_id}] Step '{step_name}' failed on attempt {attempt}/{max_attempts}: {e}; retrying..."
                    )
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)
                else:
                    logging.exception(
                        f"[{instance_id}] Step '{step_name}' failed after {max_attempts} attempts: {e}"
                    )

        return last_result

    def _load_completed_instance_ids(self) -> set:
        """
        Load instance IDs that have already been processed from the output file.
        
        Returns:
            Set of completed instance IDs (empty set if file doesn't exist or on error)
        """
        completed_ids = set()
        if os.path.exists(self.current_output_path):
            try:
                with open(self.current_output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            instance_id = record.get("instance_id")
                            if instance_id:
                                completed_ids.add(instance_id)
                logging.info(f"Found {len(completed_ids)} completed samples from previous run")
            except Exception as e:
                logging.warning(f"Failed to load completed IDs from {self.current_output_path}: {e}")
        return completed_ids

    def _filter_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out already completed samples from the dataset.
        
        Args:
            dataset: Original dataset (list of samples)
            
        Returns:
            Filtered dataset with only unprocessed samples
        """
        if not self.completed_ids:
            return dataset
        
        filtered = [sample for sample in dataset if sample.get("instance_id") not in self.completed_ids]
        logging.info(f"Filtered dataset: {len(dataset)} -> {len(filtered)} samples (skipped {len(self.completed_ids)} completed)")
        return filtered

    async def _call_llm_and_parse(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM and automatically parse JSON response.

        Args:
            system_prompt: System-level instructions for the LLM
            user_prompt: User-level task description or query
            temperature: Sampling temperature (0.0 = deterministic)
            
        Returns:
            Parsed JSON as dictionary, or None if call/parsing failed
        """
        response = await self.agent.call_llm(system_prompt, user_prompt, temperature=temperature)
        try:
            parsed_result = parse_json_output(response)
            return parsed_result
        except Exception as e:
            # Parse failures are expected intermittently and should be retried by caller.
            logging.warning(f"Failed to parse LLM JSON output, will rely on retry flow: {e}")
            return None

    async def _log_prompt_and_response(
        self, 
        instance_id: str, 
        step: str, 
        system_prompt: str, 
        user_prompt: str, 
        response: Any
    ):
        """
        Log prompts and responses to instance-specific debug file.
        
        Args:
            instance_id: Instance ID for filename
            step: Step name (e.g., 'extract_query', 'predict')
            system_prompt: System prompt sent to LLM
            user_prompt: User prompt sent to LLM
            response: Response received from LLM
        """
        if not self.debug_prompts_dir:
            return
        
        try:
            async with self.debug_lock:
                debug_file = os.path.join(self.debug_prompts_dir, f"{instance_id}.txt")
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Step: {step}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"[System Prompt]\n{system_prompt}\n\n")
                    f.write(f"[User Prompt]\n{user_prompt}\n\n")
                    f.write(f"[Response]\n{json.dumps(response, indent=2, ensure_ascii=False) if response else 'None'}\n")
        except Exception:
            pass
    
    @abstractmethod
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate evaluation metrics from results.
        
        Args:
            results: List of all evaluation results
            
        Returns:
            Dictionary of calculated metrics
        """
        pass
    
    def log_metrics(self, results: List[Dict[str, Any]]):
        """
        Log evaluation metrics.
        
        Args:
            results: List of all evaluation results
        """
        stats = self.calculate_metrics(results)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"\n=== Evaluation Metrics ===")
        for k, v in stats.items():
            logging.info(f"  {k.capitalize():15s}: {v:.4f}")
        logging.info(f"{'='*80}\n")

    async def _retrieve_from_kb(
        self, 
        query: Dict[str, Any], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant entries from knowledge base.
        
        Args:
            query: Query object for retrieval
            top_k: Maximum number of entries to retrieve
            
        Returns:
            List of retrieved KB entries (empty if KB unavailable)
            
        Raises:
            ValueError: If knowledge base is not available
        """
        if not hasattr(self.agent, 'kb') or self.agent.kb is None:
            raise ValueError("Knowledge base is not available")
        
        kb_entries = await self.agent.retrieve_memory(query, top_k)
        return kb_entries if kb_entries else []
    
    @abstractmethod
    def _build_result_record(
        self, 
        sample: Dict[str, Any], 
        prediction: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build a result record from sample and prediction.
        
        Args:
            sample: Original sample data
            prediction: Prediction/output from processing
            **kwargs: Additional context
            
        Returns:
            Formatted result record ready for storage
        """
        pass

    def _write_result(self, result: Dict[str, Any]):
        """
        Write a single result to output file immediately.
        
        Args:
            result: Result record to write
        """
        self._attach_token_usage(result)

        if self.current_output_path is None:
            return
        
        try:
            with self._write_lock:
                with open(self.current_output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception:
            pass

    @staticmethod
    def _normalise_token_stats(stats: Dict[str, Any]) -> Dict[str, int]:
        """Convert internal/OpenAI token names to the shared result schema."""
        input_tokens = int(
            stats.get("input_tokens", stats.get("total_prompt_tokens", 0)) or 0
        )
        cached_tokens = int(
            stats.get("cached_tokens", stats.get("total_cached_tokens", 0)) or 0
        )
        output_tokens = int(
            stats.get("output_tokens", stats.get("total_completion_tokens", 0)) or 0
        )
        total_tokens = int(stats.get("total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        api_calls = int(stats.get("api_calls", stats.get("api_call_count", 0)) or 0)
        return {
            "input_tokens": input_tokens,
            # This is a subset of input_tokens, not an additional token count.
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "api_calls": api_calls,
        }

    def _attach_token_usage(self, result: Dict[str, Any]) -> None:
        """Attach per-sample and cumulative LLM usage to a result record."""
        llm = getattr(self.agent, "llm", None)
        if llm is None or not getattr(llm, "tracks_token_usage", False):
            return

        scope_stats = llm.get_token_scope_stats()
        if scope_stats is not None:
            result.setdefault(
                "token_usage", self._normalise_token_stats(scope_stats)
            )

        result.setdefault(
            "cumulative_tokens",
            self._normalise_token_stats(llm.get_token_stats()),
        )

    def _reset_counters(self):
        """
        Reset success/failure counters and clear result buffers.
        
        Also resets LLM token usage statistics.
        """
        self.success_count = 0
        self.failure_count = 0
        self.results.clear()
        self.agent.llm.reset_token_stats()
    
    async def _process_with_concurrency_limit(
        self, 
        sample: Dict[str, Any],
        process_func
    ) -> bool:
        """
        Wrapper that applies concurrency limiting to sample processing.
        
        Args:
            sample: Sample to process
            process_func: Async function to process the sample
            
        Returns:
            True if processing succeeded, False otherwise
        """
        async with self._concurrency_limiter:
            llm = getattr(self.agent, "llm", None)
            scope_token = None
            if llm is not None and getattr(llm, "tracks_token_usage", False):
                scope_token = llm.start_token_scope()
            try:
                return await process_func(sample)
            finally:
                if scope_token is not None:
                    llm.end_token_scope(scope_token)

    @abstractmethod
    async def inference(self):
        """
        Run inference on the dataset using knowledge base.
        
        Returns:
            List of result records
        """
        pass
    
    @abstractmethod
    async def accumulate(self):
        """
        Accumulate experiences by processing samples and building knowledge base.
        
        Returns:
            List of result records
        """
        pass
