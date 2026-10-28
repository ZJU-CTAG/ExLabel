"""
SimpleRAG labellers: few-shot prediction over the most similar labelled
training samples.

Retrieval uses the raw task input as the key and the training annotations
(label + notes) as the value. Prediction is a single LLM call that receives
the retrieved annotated examples.
"""

import logging
from typing import Dict, Any, Optional

from prompts.prompt import PromptUtils
from labeller.issue_labeller import IssueLabeller
from labeller.test_labeller import TestLabeller

from baseline.rag_utils import build_input_key, format_annotated_examples


class SimpleRAGIssueLabeller(IssueLabeller):
    """
    SimpleRAG baseline for issue specification evaluation.

    Instead of accumulated experiences, retrieves the most similar annotated
    problem statements from the training set and uses them as few-shot
    examples for prediction.
    """

    task_type = "issue"
    label_field = "underspecified"
    retriever = None  # set by the entry point (DenseRetriever or BM25Retriever)

    async def _predict_with_rag(
        self,
        sample: Dict[str, Any],
        kb_entries: list
    ) -> Optional[Dict[str, Any]]:
        """
        Make prediction using retrieved annotated examples as few-shot context.

        Args:
            sample: Sample data containing problem_statement
            kb_entries: Retrieved memories with raw inputs and annotations

        Returns:
            Prediction dictionary with 'underspecified' field, or None if failed
        """
        instance_id = sample.get('instance_id', 'unknown')
        try:
            examples = format_annotated_examples(kb_entries, self.task_type)

            system_prompt = PromptUtils.read_prompt("system_issue", category="baseline/simple_rag")
            user_prompt = PromptUtils.format_prompt(
                PromptUtils.read_prompt("user_issue", category="baseline/simple_rag"),
                problem_statement=sample["problem_statement"],
                annotated_examples=examples
            )
            prediction = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.7)

            await self._log_prompt_and_response(instance_id, "simple_rag_predict", system_prompt, user_prompt, prediction)

            if prediction is None or self.label_field not in prediction:
                return None

            return prediction

        except Exception as e:
            logging.exception(f"[{instance_id}] SimpleRAG prediction failed: {e}")
            return None

    async def _process_inference_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Process a single sample with SimpleRAG retrieval and prediction.

        Args:
            sample: Sample data to process

        Returns:
            True if processing succeeded, False otherwise
        """
        instance_id = sample.get("instance_id", "unknown")

        try:
            query = build_input_key(sample, self.task_type)
            if self.retriever is None:
                raise ValueError("SimpleRAG retriever is not configured")
            # Retrieval is deterministic; no retry needed
            kb_entries = await self.retriever.retrieve(query, top_k=self.rag_topk)
            if not kb_entries:
                logging.warning(f"[{instance_id}] No similar training samples retrieved")
                return False

            prediction = await self._retry_step(
                "simple_rag_predict",
                lambda: self._predict_with_rag(sample, kb_entries),
                instance_id=instance_id,
            )
            if prediction is None:
                logging.warning(f"[{instance_id}] Failed to predict with SimpleRAG")
                return False

            result = self._build_result_record(sample, prediction)
            self.results.append(result)
            self._write_result(result)
            return True

        except Exception as e:
            logging.exception(f"[{instance_id}] SimpleRAG inference failed: {e}")
            return False


class SimpleRAGTestLabeller(TestLabeller):
    """
    SimpleRAG baseline for test patch evaluation.

    Retrieves the most similar annotated (problem_statement, test_patch) pairs
    from the training set and uses them as few-shot examples for prediction.
    """

    task_type = "test"
    label_field = "reasonable"
    retriever = None  # set by the entry point (DenseRetriever or BM25Retriever)

    async def _predict_with_rag(
        self,
        sample: Dict[str, Any],
        kb_entries: list
    ) -> Optional[Dict[str, Any]]:
        """
        Make prediction using retrieved annotated examples as few-shot context.

        Args:
            sample: Sample data containing problem_statement, test_patch, and patch
            kb_entries: Retrieved memories with raw inputs and annotations

        Returns:
            Prediction dictionary with 'reasonable' field, or None if failed
        """
        instance_id = sample.get('instance_id', 'unknown')
        try:
            examples = format_annotated_examples(kb_entries, self.task_type)

            system_prompt = PromptUtils.read_prompt("system_test", category="baseline/simple_rag")
            user_prompt = PromptUtils.format_prompt(
                PromptUtils.read_prompt("user_test", category="baseline/simple_rag"),
                problem_statement=sample["problem_statement"],
                test_patch=sample["test_patch"],
                patch=sample["patch"],
                annotated_examples=examples
            )
            prediction = await self._call_llm_and_parse(system_prompt, user_prompt, temperature=0.7)

            await self._log_prompt_and_response(instance_id, "simple_rag_predict", system_prompt, user_prompt, prediction)

            if prediction is None or self.label_field not in prediction:
                return None

            return prediction

        except Exception as e:
            logging.exception(f"[{instance_id}] SimpleRAG prediction failed: {e}")
            return None

    async def _process_inference_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Process a single sample with SimpleRAG retrieval and prediction.

        Args:
            sample: Sample data to process

        Returns:
            True if processing succeeded, False otherwise
        """
        instance_id = sample.get("instance_id", "unknown")

        try:
            query = build_input_key(sample, self.task_type)
            if self.retriever is None:
                raise ValueError("SimpleRAG retriever is not configured")
            # Retrieval is deterministic; no retry needed
            kb_entries = await self.retriever.retrieve(query, top_k=self.rag_topk)
            if not kb_entries:
                logging.warning(f"[{instance_id}] No similar training samples retrieved")
                return False

            prediction = await self._retry_step(
                "simple_rag_predict",
                lambda: self._predict_with_rag(sample, kb_entries),
                instance_id=instance_id,
            )
            if prediction is None:
                logging.warning(f"[{instance_id}] Failed to predict with SimpleRAG")
                return False

            result = self._build_result_record(sample, prediction)
            self.results.append(result)
            self._write_result(result)
            return True

        except Exception as e:
            logging.exception(f"[{instance_id}] SimpleRAG inference failed: {e}")
            return False
