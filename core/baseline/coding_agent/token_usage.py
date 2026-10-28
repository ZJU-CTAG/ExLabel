"""Token-usage tracking shared by coding-agent labellers."""

from contextvars import ContextVar, Token
import logging
from typing import Any, Dict, Optional


def _usage_value(usage: Any, field: str) -> int:
    """Read one count from a CLI usage mapping or OpenAI-style detail object."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(field, 0)
    else:
        value = getattr(usage, field, 0)
    return int(value or 0)


def _cached_tokens(usage: Any) -> int:
    """Read cache-read tokens from Claude, Codex, and OpenAI-style usage."""
    for details_field in ("prompt_tokens_details", "input_tokens_details"):
        if isinstance(usage, dict):
            details = usage.get(details_field)
        else:
            details = getattr(usage, details_field, None)
        if details is not None:
            value = _usage_value(details, "cached_tokens")
            if value:
                return value

    for field in ("cached_tokens", "cached_input_tokens", "cache_read_input_tokens"):
        value = _usage_value(usage, field)
        if value:
            return value
    return 0


def _cache_creation_tokens(usage: Any) -> int:
    """Read cache-write tokens when a provider reports them separately."""
    for field in ("cache_creation_tokens", "cache_creation_input_tokens"):
        value = _usage_value(usage, field)
        if value:
            return value
    return 0


def empty_coding_agent_usage() -> Dict[str, int]:
    """Return the shared result schema with zero usage."""
    return {
        "input_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }


def normalise_coding_agent_usage(
    usage: Dict[str, Any], agent_name: Optional[str]
) -> Dict[str, int]:
    """Convert a Claude/Codex CLI usage envelope to the shared result schema.

    Claude reports cache reads and cache creation separately from
    ``input_tokens``.  Codex reports cached input as a subset of
    ``input_tokens``.  The individual category fields are therefore retained
    for price calculation instead of attempting to merge differently priced
    categories.
    """
    input_tokens = _usage_value(usage, "input_tokens")
    if input_tokens == 0:
        input_tokens = _usage_value(usage, "prompt_tokens")
    cached_tokens = _cached_tokens(usage)
    cache_creation_tokens = _cache_creation_tokens(usage)
    output_tokens = _usage_value(usage, "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens")

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
        # Claude exposes cached categories outside input_tokens, while Codex
        # counts cached input within input_tokens.
        if agent_name == "claude":
            total_tokens += cached_tokens + cache_creation_tokens

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "api_calls": 1,
    }


def add_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
    """Add a normalised usage record to an existing total in place."""
    for field, value in usage.items():
        total[field] = total.get(field, 0) + value


class CodingAgentUsageTracker:
    """Per-sample and cumulative usage tracking for external coding agents."""

    def _coding_agent_scope_var(self) -> ContextVar[Optional[Dict[str, int]]]:
        scope = getattr(self, "_coding_agent_token_scope", None)
        if scope is None:
            scope = ContextVar(f"coding_agent_token_scope_{id(self)}", default=None)
            self._coding_agent_token_scope = scope
        return scope

    def _start_coding_agent_token_scope(self) -> Token:
        return self._coding_agent_scope_var().set(empty_coding_agent_usage())

    def _get_coding_agent_token_scope_stats(self) -> Optional[Dict[str, int]]:
        stats = self._coding_agent_scope_var().get()
        return dict(stats) if stats is not None else None

    def _end_coding_agent_token_scope(self, token: Token) -> Dict[str, int]:
        stats = self._get_coding_agent_token_scope_stats() or empty_coding_agent_usage()
        self._coding_agent_scope_var().reset(token)
        return stats

    def _accumulate_tokens(self, usage: Optional[Dict[str, Any]]) -> None:
        """Record one coding-agent session, including cache-read categories."""
        if not usage:
            return

        normalised = normalise_coding_agent_usage(
            usage, getattr(self, "agent_name", None)
        )
        if not hasattr(self, "_coding_agent_token_stats"):
            self._coding_agent_token_stats = empty_coding_agent_usage()
        add_usage(self._coding_agent_token_stats, normalised)

        scope = self._coding_agent_scope_var().get()
        if scope is not None:
            add_usage(scope, normalised)

    async def _process_with_concurrency_limit(self, sample, process_func) -> bool:
        """Extend the base sample scope with coding-agent usage isolation."""
        token = self._start_coding_agent_token_scope()
        try:
            return await super()._process_with_concurrency_limit(sample, process_func)
        finally:
            self._end_coding_agent_token_scope(token)

    async def inference(self):
        """Run inference and report the external coding-agent token totals."""
        results = await super().inference()
        stats = getattr(self, "_coding_agent_token_stats", empty_coding_agent_usage())
        logging.info("Coding-agent token usage statistics:")
        logging.info("  Total Input Tokens:          %s", f"{stats['input_tokens']:,}")
        logging.info("  Total Cached Tokens:         %s", f"{stats['cached_tokens']:,}")
        logging.info(
            "  Total Cache Creation Tokens: %s",
            f"{stats['cache_creation_tokens']:,}",
        )
        logging.info("  Total Output Tokens:         %s", f"{stats['output_tokens']:,}")
        logging.info("  Total Tokens:                %s", f"{stats['total_tokens']:,}")
        logging.info("  Total API Calls:             %s", stats["api_calls"])
        return results

    def _write_result(self, result: Dict[str, Any]):
        """Persist coding-agent usage without hiding any plain-LLM usage."""
        sample_usage = self._get_coding_agent_token_scope_stats()
        totals = getattr(self, "_coding_agent_token_stats", empty_coding_agent_usage())
        if sample_usage is not None:
            result.setdefault("coding_agent_token_usage", sample_usage)
        result.setdefault("cumulative_coding_agent_tokens", dict(totals))

        llm = getattr(getattr(self, "agent", None), "llm", None)
        if llm is None or not getattr(llm, "tracks_token_usage", False):
            # The standalone coding-agent baseline has no plain LLM.  Keep the
            # shared token_usage/cumulative_tokens fields populated for it.
            if sample_usage is not None:
                result.setdefault("token_usage", sample_usage)
            result.setdefault("cumulative_tokens", dict(totals))

        return super()._write_result(result)
