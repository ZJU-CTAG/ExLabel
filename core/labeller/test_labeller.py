import logging
import json
import asyncio
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score

from prompts.prompt import PromptUtils
from labeller.base import BaseLabeller


class TestLabeller(BaseLabeller):
    """
    Labeller for test patch evaluation with knowledge base support.
    
    Handles both inference (using KB) and accumulation (building KB) modes
    for labeling test patches as reasonable or unreasonable.
    
    Prediction logic is implemented here, while KB building logic is delegated
    to the base class and LLM prompts for clean separation of concerns.
    """

    task_type = "test"
    
    def __init__(
        self, 
        agent, 
        dataset,
        output_path: Optional[str] = None,
        max_concurrent: int = 100, 
        ablation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize test labeller.
        
        Args:
            agent: Agent instance with LLM and knowledge base
            dataset: Dataset of test labeling samples
            output_path: Path to save results as JSONL file (None to skip)
            max_concurrent: Maximum concurrent tasks
            ablation_config: Configuration for ablation study variants
        """
        super().__init__(agent, dataset, output_path, max_concurrent, ablation_config)
    
    async def _extract_issue_query(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract key issues from problem statement and test patch as a structured query.
        
        Args:
            sample: Sample data containing problem_statement and test_patch
            
        Returns:
            Structured query dictionary, or None if extraction failed
        """
        instance_id = sample.get('instance_id', 'unknown')
        try:
            system_prompt = PromptUtils.read_prompt("issue_extraction_system", category="test_labeller")
            user_prompt = PromptUtils.format_prompt(
                PromptUtils.read_prompt("issue_extraction_user", category="test_labeller"),
                problem_statement=sample["problem_statement"],
                test_patch=sample["test_patch"]
            )
            query = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.5)
            
            await self._log_prompt_and_response(instance_id, "extract_issue_query", system_prompt, user_prompt, query)
            
            return query
            
        except Exception as e:
            logging.exception(f"[{instance_id}] Issue query extraction failed: {e}")
            return None
    
    async def _rerank_kb_entries(
        self,
        sample: Dict[str, Any],
        kb_entries: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Rerank KB entries by relevance and select the most relevant ones.

        Args:
            sample: Sample data containing problem_statement and test_patch
            kb_entries: List of candidate KB entries to rerank

        Returns:
            List of the most relevant KB entries (up to rerank_topk, each with reason),
            or None if reranking failed
        """
        instance_id = sample.get('instance_id', 'unknown')
        try:
            if not kb_entries:
                logging.warning(f"[{instance_id}] No KB entries provided for reranking")
                return None

            if not self.use_reranking:
                selected_entries = [entry.copy() for entry in kb_entries[:self.rerank_topk]]
                for selected_entry in selected_entries:
                    selected_entry["reason"] = "No reranking (ablation study)"
                return selected_entries

            experiences, candidate_entries = self._prepare_rerank_candidates(kb_entries)

            if self.rerank_topk > 1:
                system_prompt = PromptUtils.read_prompt("rerank_system_topk", category="test_labeller")
                user_prompt = PromptUtils.format_prompt(
                    PromptUtils.read_prompt("rerank_user_topk", category="test_labeller"),
                    problem_statement=sample["problem_statement"],
                    test_patch=sample["test_patch"],
                    experiences=experiences,
                    topk=self.rerank_topk
                )
            else:
                system_prompt = PromptUtils.read_prompt("rerank_system", category="test_labeller")
                user_prompt = PromptUtils.format_prompt(
                    PromptUtils.read_prompt("rerank_user", category="test_labeller"),
                    problem_statement=sample["problem_statement"],
                    test_patch=sample["test_patch"],
                    experiences=experiences
                )

            rerank_result = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.5)

            await self._log_prompt_and_response(instance_id, "rerank_kb_entries", system_prompt, user_prompt, rerank_result)

            if rerank_result is None:
                return None

            if self.rerank_topk > 1:
                selected_ids = rerank_result.get("candidate_ids")
                reason = rerank_result.get("reason", "")
                if not selected_ids or not isinstance(selected_ids, list):
                    return None
            else:
                selected_id = rerank_result.get("candidate_id")
                if not selected_id:
                    return None
                selected_ids = [selected_id]
                reason = rerank_result.get("reason", "")

            selected_entries = []
            for selected_id in selected_ids[:self.rerank_topk]:
                entry = candidate_entries.get(selected_id)
                if entry is not None:
                    selected_entry = entry.copy()
                    selected_entry["reason"] = reason
                    selected_entries.append(selected_entry)

            return selected_entries if selected_entries else None

        except Exception as e:
            logging.exception(f"[{instance_id}] KB reranking failed: {e}")
            return None
    
    def _build_eval_prompts(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """
        Build the eval prompts used when no KB experience is available.

        Args:
            sample: Sample data containing problem_statement, test_patch, and patch

        Returns:
            (system prompt, user prompt) for the plain eval templates
        """
        system_prompt = PromptUtils.read_prompt("eval_system", category="test_labeller")
        user_prompt = PromptUtils.format_prompt(
            PromptUtils.read_prompt("eval_user", category="test_labeller"),
            problem_statement=sample["problem_statement"],
            test_patch=sample["test_patch"],
            patch=sample["patch"]
        )
        return system_prompt, user_prompt

    def _build_eval_kb_prompts(
        self,
        sample: Dict[str, Any],
        kb_entries: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Build the eval prompts that integrate retrieved KB experiences.

        Args:
            sample: Sample data containing problem_statement, test_patch, and patch
            kb_entries: KB entries with experiences to guide prediction

        Returns:
            (system prompt, user prompt) for the experience-guided eval templates
        """
        experiences = []
        for kb_entry in kb_entries:
            experience = {
                "perspective": kb_entry.get("perspective", ""),
            }
            if not self.mask_experience_field:
                experience["experience"] = kb_entry.get("experience", "")
            if "mistakes" in kb_entry and not self.mask_mistakes_field:
                experience["mistakes"] = kb_entry["mistakes"]
            experiences.append(experience)

        if len(experiences) > 1:
            system_prompt = PromptUtils.read_prompt("eval_kb_system_topk", category="test_labeller")
            experience_str = json.dumps(experiences, indent=2, ensure_ascii=False)
        else:
            system_prompt = PromptUtils.read_prompt("eval_kb_system", category="test_labeller")
            experience_str = json.dumps(experiences[0], indent=2, ensure_ascii=False)

        user_prompt = PromptUtils.format_prompt(
            PromptUtils.read_prompt("eval_kb_user", category="test_labeller"),
            problem_statement=sample["problem_statement"],
            test_patch=sample["test_patch"],
            patch=sample["patch"],
            experience=experience_str
        )
        return system_prompt, user_prompt

    async def _predict_without_kb(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make prediction without KB (fallback when KB is empty).
        
        Args:
            sample: Sample data containing problem_statement, test_patch, and patch
            
        Returns:
            Prediction dictionary with 'reasonable' field, or None if failed
        """
        instance_id = sample.get('instance_id', 'unknown')
        try:
            system_prompt, user_prompt = self._build_eval_prompts(sample)
            prediction = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.7)
            
            await self._log_prompt_and_response(instance_id, "predict_fallback", system_prompt, user_prompt, prediction)
            
            if prediction is None or "reasonable" not in prediction:
                return None
            
            return prediction
            
        except Exception as e:
            logging.exception(f"[{instance_id}] Fallback prediction failed: {e}")
            return None
    
    async def _predict_with_kb(
        self,
        sample: Dict[str, Any],
        kb_entries: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Make prediction using knowledge base experiences.

        Args:
            sample: Sample data containing problem_statement, test_patch, and patch
            kb_entries: KB entries with experiences to guide prediction

        Returns:
            Prediction dictionary with 'reasonable' field, or None if failed
        """
        instance_id = sample.get('instance_id', 'unknown')
        kb_ids = [entry.get('instance_id', 'unknown') for entry in kb_entries]

        try:
            system_prompt, user_prompt = self._build_eval_kb_prompts(sample, kb_entries)
            prediction = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.7)

            await self._log_prompt_and_response(instance_id, "predict_with_kb", system_prompt, user_prompt, prediction)

            if prediction is None or "reasonable" not in prediction:
                return None

            return prediction

        except Exception as e:
            logging.exception(f"[{instance_id}] KB prediction failed (entries: {kb_ids}): {e}")
            return None

    async def _extract_experience(
        self, 
        sample: Dict[str, Any], 
        prediction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract experience/lessons from prediction outcome.
        
        Args:
            sample: Sample data with ground truth
            prediction: Model's prediction result
            
        Returns:
            Experience dictionary with perspective and experience fields, or None if failed
        """
        instance_id = sample.get('instance_id', 'unknown')
        
        try:
            is_correct = prediction["reasonable"] == sample["reasonable"]
            
            prompt_type = "successful_experience" if is_correct else "failed_experience"
            system_prompt = PromptUtils.read_prompt(f"{prompt_type}_system", category="test_labeller")
            user_prompt = PromptUtils.format_prompt(
                PromptUtils.read_prompt(f"{prompt_type}_user", category="test_labeller"),
                problem_statement=sample["problem_statement"],
                test_patch=sample["test_patch"],
                reasonable=str(sample["reasonable"]).lower(),
                notes="\n".join(sample.get("notes", [])),
                prediction=json.dumps(prediction, indent=2, ensure_ascii=False)
            )
            
            experience = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.7)
            
            await self._log_prompt_and_response(instance_id, "extract_experience", system_prompt, user_prompt, experience)
            
            if experience is None:
                return None
            
            required_fields = ["perspective", "experience"]
            missing = [f for f in required_fields if f not in experience]
            if missing:
                return None
            
            return experience
            
        except Exception as e:
            logging.exception(f"[{instance_id}] Experience extraction failed: {e}")
            return None

    async def _generate_kb_operations(
        self,
        sample: Dict[str, Any],
        prediction: Dict[str, Any],
        kb_entry: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate KB operations based on current prediction and existing KB entry.
        
        Args:
            sample: Sample data with ground truth
            prediction: Model's prediction result
            kb_entry: Existing KB entry that was used
            
        Returns:
            List of KB operations (insert/modify/delete), or None if generation failed
        """
        instance_id = sample.get("instance_id", "unknown")
        
        try:
            experience = {
                "id": kb_entry.get("id", ""),
                "perspective": kb_entry.get("perspective", ""),
                "experience": kb_entry.get("experience", "")
            }
            if "mistakes" in kb_entry:
                experience["mistakes"] = kb_entry["mistakes"]
            
            system_prompt = PromptUtils.read_prompt("experience_system", category="test_labeller")
            user_prompt = PromptUtils.format_prompt(
                PromptUtils.read_prompt("experience_user", category="test_labeller"),
                problem_statement=sample.get("problem_statement", ""),
                test_patch=sample.get("test_patch", ""),
                reasonable=str(sample.get("reasonable")).lower(),
                notes="\n".join(sample.get("notes", [])),
                prediction=json.dumps(prediction, indent=2, ensure_ascii=False),
                experience=json.dumps(experience, indent=2, ensure_ascii=False)
            )            
            operations_dict = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.7)
            if operations_dict is None:
                return None
            
            operations = operations_dict.get("operations", [])
            if not isinstance(operations, list):
                return None
            
            return operations
            
        except Exception as e:
            logging.exception(f"[{instance_id}] Failed to generate KB operations: {e}")
            return None

    def _build_result_record(
        self, 
        sample: Dict[str, Any], 
        prediction: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build result record with prediction and annotation.
        
        Args:
            sample: Original sample data
            prediction: Model's prediction
            **kwargs: Additional context (unused)
            
        Returns:
            Result record with instance_id, prediction, and annotation
        """
        record = {
            "instance_id": sample.get("instance_id", ""),
            "prediction": prediction,
            "annotation": {
                "reasonable": sample.get("reasonable", None),
                "notes": sample.get("notes", []),
            },
        }
        return record

    def calculate_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute precision/recall/accuracy for reasonableness (positive = unreasonable)."""
        y_true: List[bool] = []
        y_pred: List[bool] = []

        for data in records:
            prediction = data.get("prediction")
            annotation = data.get("annotation")

            if not isinstance(prediction, dict) or not isinstance(annotation, dict):
                continue

            pred_val = prediction.get("reasonable")
            true_val = annotation.get("reasonable")

            if isinstance(pred_val, bool) and isinstance(true_val, bool):
                y_true.append(true_val)
                y_pred.append(pred_val)

        if not y_true:
            return {"precision": 0.0, "recall": 0.0, "accuracy": 0.0}

        precision = precision_score(y_true, y_pred, pos_label=False, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=False, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

    async def _process_inference_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Process a single sample for inference using KB.
        
        Args:
            sample: Sample data to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        instance_id = sample.get("instance_id", "unknown")
        
        try:
            if not self.use_experience_bank:
                prediction = await self._retry_step(
                    "predict_without_kb",
                    lambda: self._predict_without_kb(sample),
                    instance_id=instance_id,
                )
                if prediction is None:
                    logging.warning(f"[{instance_id}] Failed to predict without KB")
                    return False
                
                result = self._build_result_record(sample, prediction)
                self.results.append(result)
                self._write_result(result)
                return True
            
            query = await self._retry_step(
                "extract_issue_query",
                lambda: self._extract_issue_query(sample),
                instance_id=instance_id,
            )
            if query is None:
                logging.warning(f"[{instance_id}] Failed to extract query")
                return False
            
            kb_entries = await self._retry_step(
                "retrieve_from_kb",
                lambda: self._retrieve_from_kb(query, top_k=10),
                instance_id=instance_id,
                retry_on_empty=True,
            )
            if not kb_entries:
                logging.warning(f"[{instance_id}] No KB entries retrieved")
                return False

            selected_entries = await self._retry_step(
                "rerank_kb_entries",
                lambda: self._rerank_kb_entries(sample, kb_entries),
                instance_id=instance_id,
            )
            if not selected_entries:
                logging.warning(f"[{instance_id}] Failed to rerank KB entries")
                return False

            prediction = await self._retry_step(
                "predict_with_kb",
                lambda: self._predict_with_kb(sample, selected_entries),
                instance_id=instance_id,
            )
            if prediction is None:
                logging.warning(f"[{instance_id}] Failed to predict with KB")
                return False
                        
            result = self._build_result_record(sample, prediction)
            self.results.append(result)
            self._write_result(result)
            return True
            
        except Exception as e:
            logging.exception(f"[{instance_id}] Inference processing failed: {e}")
            return False
    
    async def _process_accumulation_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Process a single sample for accumulation (building KB).
        
        Args:
            sample: Sample data to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        instance_id = sample.get("instance_id", "unknown")
        
        try:
            query = await self._retry_step(
                "extract_issue_query",
                lambda: self._extract_issue_query(sample),
                instance_id=instance_id,
            )
            if query is None:
                logging.warning(f"[{instance_id}] Failed to extract query")
                return False
            
            kb_entries = await self._retrieve_from_kb(query, top_k=10)

            selected_entries = None
            if kb_entries:
                selected_entries = await self._retry_step(
                    "rerank_kb_entries",
                    lambda: self._rerank_kb_entries(sample, kb_entries),
                    instance_id=instance_id,
                )
                if selected_entries:
                    prediction = await self._retry_step(
                        "predict_with_kb",
                        lambda: self._predict_with_kb(sample, selected_entries),
                        instance_id=instance_id,
                    )
                    if prediction is None:
                        logging.debug(f"[{instance_id}] KB prediction failed, falling back to no-KB prediction")
                        prediction = await self._retry_step(
                            "predict_without_kb",
                            lambda: self._predict_without_kb(sample),
                            instance_id=instance_id,
                        )
                        selected_entries = None
                else:
                    logging.debug(f"[{instance_id}] Reranking failed, using no-KB prediction")
                    prediction = await self._retry_step(
                        "predict_without_kb",
                        lambda: self._predict_without_kb(sample),
                        instance_id=instance_id,
                    )
            else:
                logging.debug(f"[{instance_id}] No KB entries, using no-KB prediction")
                prediction = await self._retry_step(
                    "predict_without_kb",
                    lambda: self._predict_without_kb(sample),
                    instance_id=instance_id,
                )

            if prediction is None:
                logging.warning(f"[{instance_id}] All prediction methods failed")
                return False

            result = self._build_result_record(sample, prediction)
            self.results.append(result)
            self._write_result(result)

            if selected_entries is None:
                experience = await self._retry_step(
                    "extract_experience",
                    lambda: self._extract_experience(sample, prediction),
                    instance_id=instance_id,
                )
                if experience is None:
                    logging.warning(f"[{instance_id}] Failed to extract experience")
                    return False

                kb_value = experience.copy()
                success = await self.agent.add_memory(query, kb_value)
                self.operation_dict["insert"] +=1
            else:
                # KB operations still target the single most relevant entry
                kb_entry = selected_entries[0]
                operations = await self._retry_step(
                    "generate_kb_operations",
                    lambda: self._generate_kb_operations(sample, prediction, kb_entry),
                    instance_id=instance_id,
                )
                if operations is None:
                    logging.warning(f"[{instance_id}] Failed to generate KB operations")
                    return False

                has_modify = any(op.get("type") == "modify" for op in operations)
                has_insert = any(op.get("type") == "insert" for op in operations)     
                if has_modify and not has_insert:
                    modify_op = next((op for op in operations if op.get("type") == "modify"), None)
                    if modify_op and "content" in modify_op:
                        modify_op["key"] = query

                for operation in operations:
                    if operation.get("type") == "insert":
                        operation["key"] = query
                    
                    success = await self.agent.operate(operation)
                    self.operation_dict[operation.get("type", "unknown")] +=1
                        
            return True
        except Exception as e:
            logging.exception(f"[{instance_id}] Accumulation failed: {e}")
            return False
    
    async def inference(self):
        """
        Run inference on the dataset using knowledge base.
        
        Returns:
            List of result records
        """
        logging.info(f"Starting test labeller inference: {len(self.dataset)} samples")
        
        tasks = [
            self._process_with_concurrency_limit(sample, self._process_inference_sample)
            for sample in self.dataset
        ]
        
        with tqdm(
            total=len(tasks),
            desc=f"inferencing",
            dynamic_ncols=True,
        ) as pbar:
            for task in asyncio.as_completed(tasks):
                success = await task
                if success:
                    self.success_count += 1
                else:
                    self.failure_count += 1
                pbar.update(1)
        
        self.log_metrics(self.results)
        logging.info(
            f"Test labeller inference completed: Success={self.success_count}, "
            f"Failures={self.failure_count}, Total={len(self.dataset)}"
        )
        
        token_stats = self.agent.llm.get_token_stats()
        logging.info(f"\n{'='*80}")
        logging.info(f"Final Token Usage Statistics:")
        logging.info(f"  Total Prompt Tokens:     {token_stats['total_prompt_tokens']:,}")
        logging.info(f"  Total Cached Tokens:     {token_stats['total_cached_tokens']:,}")
        logging.info(f"  Total Completion Tokens: {token_stats['total_completion_tokens']:,}")
        logging.info(f"  Total Tokens:            {token_stats['total_tokens']:,}")
        logging.info(f"  Total API Calls:         {token_stats['api_call_count']}")
        logging.info(f"{'='*80}\n")
                
        return self.results
    
    async def accumulate(self):
        """
        Accumulate experiences by processing samples and building knowledge base.
        
        Returns:
            List of result records
        """
        logging.info(f"Starting test labeller accumulation: {len(self.dataset)} samples")
        
        tasks = [
            self._process_with_concurrency_limit(sample, self._process_accumulation_sample)
            for sample in self.dataset
        ]
        
        with tqdm(
            total=len(tasks),
            desc=f"accumulating",
            dynamic_ncols=True,
        ) as pbar:
            for task in asyncio.as_completed(tasks):
                success = await task
                if success:
                    self.success_count += 1
                else:
                    self.failure_count += 1
                pbar.update(1)
        
        logging.info(
            f"Test labeller accumulation completed: Success={self.success_count}, "
            f"Failures={self.failure_count}, Total={len(self.dataset)}"
        )
        
        token_stats = self.agent.llm.get_token_stats()
        logging.info(f"\n{'='*80}")
        logging.info(f"Final Token Usage Statistics:")
        logging.info(f"  Total Prompt Tokens:     {token_stats['total_prompt_tokens']:,}")
        logging.info(f"  Total Cached Tokens:     {token_stats['total_cached_tokens']:,}")
        logging.info(f"  Total Completion Tokens: {token_stats['total_completion_tokens']:,}")
        logging.info(f"  Total Tokens:            {token_stats['total_tokens']:,}")
        logging.info(f"  Total API Calls:         {token_stats['api_call_count']}")
        logging.info(f"{'='*80}\n")
        
        if hasattr(self.agent, 'kb') and self.agent.kb:
            self.agent.kb.save()
        
        return self.results

