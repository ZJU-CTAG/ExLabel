"""Vanilla labellers with baseline-specific final-prediction prompts."""

from typing import Any, Dict, Tuple

from labeller.test_labeller import TestLabeller
from prompts.prompt import PromptUtils


class VanillaTestLabeller(TestLabeller):
    """Test labeller that uses the deliberately minimal vanilla prompt."""

    def _build_eval_prompts(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = PromptUtils.read_prompt(
            "system_test", category="baseline/vanilla"
        )
        user_prompt = PromptUtils.format_prompt(
            PromptUtils.read_prompt("eval_user", category="test_labeller"),
            problem_statement=sample["problem_statement"],
            test_patch=sample["test_patch"],
            patch=sample["patch"],
        )
        return system_prompt, user_prompt
