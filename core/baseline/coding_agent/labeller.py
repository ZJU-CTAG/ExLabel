"""
Coding agent labellers: an external coding agent (claude code / codex) acts
as the labeller with access to the target repository.

Every sample gets a fresh playground worktree at its `base_commit`; the task
prompt (core/prompts/baseline/coding_agent/<task>.md) is piped to the agent
CLI, and the agent's final message must be a strict JSON object carrying the
task label (`underspecified` / `reasonable`).

Result records, metrics, retry and resume behaviour are inherited from the
parent labellers, so the JSONL output matches the other baselines exactly.
"""

import logging
from typing import Any, Dict, Optional

from prompts.prompt import PromptUtils
from utils.json_parser import parse_json_output
from labeller.issue_labeller import IssueLabeller
from labeller.test_labeller import TestLabeller

from baseline.coding_agent.cli import run_coding_agent
from baseline.coding_agent.token_usage import CodingAgentUsageTracker


def build_agent_prompt(task: str, sample: Dict[str, Any]) -> str:
    """
    Render the coding agent task prompt for one sample.

    Args:
        task: Task type ('issue' or 'test')
        sample: Dataset sample with the raw input fields

    Returns:
        The full prompt text to pipe to the agent CLI
    """
    template = PromptUtils.read_prompt(task, category="baseline/coding_agent")
    kwargs = {"problem_statement": sample.get("problem_statement", "")}
    if task == "test":
        kwargs["patch"] = sample.get("patch", "")
        kwargs["test_patch"] = sample.get("test_patch", "")
    return PromptUtils.format_prompt(template, **kwargs)


class CodingAgentMixin(CodingAgentUsageTracker):
    """
    Shared wiring for the coding agent labellers.

    Configuration attributes are injected by core/baseline/run_coding_agent.py.
    """

    task_type: Optional[str] = None        # 'issue' | 'test'
    agent_name: Optional[str] = None       # 'claude' | 'codex' | 'fake'
    agent_model: Optional[str] = None
    agent_timeout: float = 900.0
    playground = None                      # baseline.playground.Playground
    keep_worktrees: bool = False

    async def _run_agent_and_parse(
        self,
        prompt: str,
        worktree_path,
        instance_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Run one agent session and parse its final message.

        Executed as a retried step: agent failures, timeouts and unparseable
        output all return None so _retry_step re-runs the whole session.

        Returns:
            Prediction dict carrying a boolean task label, or None if failed
        """
        final_text, usage = await run_coding_agent(
            agent=self.agent_name,
            label_field=self.label_field,
            prompt=prompt,
            workdir=str(worktree_path),
            instance_id=instance_id,
            out_dir=self.debug_prompts_dir or ".",
            model=self.agent_model,
            timeout_seconds=self.agent_timeout,
        )
        self._accumulate_tokens(usage)
        await self._log_prompt_and_response(
            instance_id,
            "coding_agent_run",
            f"agent={self.agent_name} model={self.agent_model or 'default'} cwd={worktree_path}",
            prompt,
            final_text,
        )
        if not final_text:
            return None

        try:
            prediction = parse_json_output(final_text)
        except Exception as e:
            logging.warning(f"[{instance_id}] Agent output is not valid JSON: {e}")
            return None

        if not isinstance(prediction, dict) or not isinstance(prediction.get(self.label_field), bool):
            logging.warning(f"[{instance_id}] Agent output is missing boolean '{self.label_field}'")
            return None

        return prediction

    async def _process_inference_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Process a single sample: create its worktree, run the coding agent,
        parse the label, and persist the result.

        Args:
            sample: Dataset sample (requires repo, base_commit and task inputs)

        Returns:
            True if processing succeeded, False otherwise
        """
        instance_id = sample.get("instance_id", "unknown")
        repo = sample.get("repo")
        base_commit = sample.get("base_commit")

        try:
            if self.playground is None or not self.agent_name:
                raise ValueError(
                    "Coding agent labeller is not configured (playground/agent_name missing)"
                )
            if not repo or not base_commit:
                logging.warning(
                    f"[{instance_id}] Sample is missing repo/base_commit; "
                    f"the coding agent baseline requires both"
                )
                return False

            worktree_path = await self.playground.create_worktree(
                repo, base_commit, instance_id
            )
            try:
                prompt = build_agent_prompt(self.task_type, sample)
                prediction = await self._retry_step(
                    "coding_agent_run",
                    lambda: self._run_agent_and_parse(prompt, worktree_path, instance_id),
                    instance_id=instance_id,
                )
            finally:
                if not self.keep_worktrees:
                    await self.playground.discard_worktree(repo, instance_id)

            if prediction is None:
                logging.warning(f"[{instance_id}] Coding agent failed to produce a valid prediction")
                return False

            result = self._build_result_record(sample, prediction)
            self.results.append(result)
            self._write_result(result)
            return True

        except Exception as e:
            logging.exception(f"[{instance_id}] Coding agent inference failed: {e}")
            return False


class CodingAgentIssueLabeller(CodingAgentMixin, IssueLabeller):
    """Coding agent baseline for issue specification evaluation."""

    task_type = "issue"
    label_field = "underspecified"


class CodingAgentTestLabeller(CodingAgentMixin, TestLabeller):
    """Coding agent baseline for test patch evaluation."""

    task_type = "test"
    label_field = "reasonable"
