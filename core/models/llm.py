import asyncio
import logging
import os
from contextvars import ContextVar, Token
from typing import Any, Dict, Optional

from openai import AsyncOpenAI, APITimeoutError


class LLM:
    """
    Async LLM client with automatic retry logic.

    Provides asynchronous interface to language models with:
    - Automatic retry on failures
    - Timeout with temperature escalation (breaks stuck generations)
    - Token usage tracking
    - JSON response format
    """

    tracks_token_usage = True

    def __init__(
        self,
        model: str,
        base_url: str = None,
        api_key: str = None,
        max_retries: int = 3,
        use_json_mode: bool = True,
        timeout: int = 180,
        temp_step: float = 0.3,
        reasoning_effort: Optional[str] = "xhigh",
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name/identifier
            base_url: Optional custom API base URL
            api_key: API key for authentication (defaults to API_KEY from env)
            max_retries: Maximum number of attempts. Defaults to three.
            use_json_mode: Whether to use JSON response format (not all providers support this)
            timeout: Per-request timeout in seconds (default: 180). When a request exceeds this,
                it is retried with an escalated temperature (temperature + temp_step per retry)
                to break out of a stuck generation.
            temp_step: How much to raise the temperature on each timeout retry.
            reasoning_effort: Reasoning effort sent to OpenAI-compatible models.
                Defaults to ``xhigh``; set EXLABEL_LLM_REASONING_EFFORT to
                override it, or an empty string to omit the parameter.
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("API_KEY")

        if not self.api_key:
            raise ValueError("API key must be provided or set API_KEY environment variable")

        # Keep transport retries in the explicit attempt loop below so all
        # retries follow the configured limit and timeout-escalation policy.
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.max_retries = max_retries
        self.use_json_mode = use_json_mode
        self.timeout = timeout
        self.temp_step = temp_step
        self.reasoning_effort = os.getenv(
            "EXLABEL_LLM_REASONING_EFFORT", reasoning_effort or ""
        )
        self._last_call_time = 0
        self._lock = asyncio.Lock()

        self.total_prompt_tokens = 0
        # Cached tokens are already included in prompt tokens.  Keep them as a
        # separate counter so callers can apply the provider's cache-read rate
        # without counting them twice in total_tokens.
        self.total_cached_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        self._stats_lock = asyncio.Lock()
        # Each sample is processed in its own asyncio task.  A ContextVar keeps
        # per-sample usage isolated even when many requests finish concurrently.
        self._token_scope: ContextVar[Optional[Dict[str, int]]] = ContextVar(
            f"llm_token_scope_{id(self)}", default=None
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    async def _completion_request(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Make a completion request to the LLM API with retry + temperature escalation.

        On a timeout (request exceeding the client timeout), the temperature is
        raised by temp_step for the next attempt so a pathological/stuck
        generation path is abandoned. Other errors retry at the same temperature
        to keep results deterministic.

        Args:
            system_prompt: System-level instructions
            user_prompt: User query
            temperature: Sampling temperature

        Returns:
            Response content string
        """
        # Allow a global temperature override via env var (e.g., deterministic runs).
        # When set, it applies to every call regardless of the per-step default.
        base_temperature = float(os.getenv("EXLABEL_LLM_TEMPERATURE", temperature))

        timeout_escalations = 0
        last_exc = None

        for attempt in range(1, self.max_retries + 1):
            # Escalate temperature only after timeouts; keep base temp otherwise.
            current_temp = min(
                base_temperature + timeout_escalations * self.temp_step, 1.0
            )

            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": current_temp,
            }

            if self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort

            if self.use_json_mode:
                params["response_format"] = {"type": "json_object"}

            # Optional per-run guard against pathological generations. Leaving the
            # variable unset preserves the provider/model default for existing runs.
            max_tokens = os.getenv("EXLABEL_LLM_MAX_TOKENS")
            if max_tokens:
                params["max_tokens"] = int(max_tokens)

            try:
                response = await self.client.chat.completions.create(**params)

                usage = getattr(response, "usage", None)
                if usage is not None:
                    prompt_tokens = self._usage_value(usage, "prompt_tokens")
                    cached_tokens = self._cached_token_count(usage)
                    completion_tokens = self._usage_value(usage, "completion_tokens")
                    total_tokens = self._usage_value(usage, "total_tokens")
                    if total_tokens == 0:
                        total_tokens = prompt_tokens + completion_tokens

                    async with self._stats_lock:
                        self.total_prompt_tokens += prompt_tokens
                        self.total_cached_tokens += cached_tokens
                        self.total_completion_tokens += completion_tokens
                        self.total_tokens += total_tokens
                        self.api_call_count += 1

                    scope = self._token_scope.get()
                    if scope is not None:
                        scope["total_prompt_tokens"] += prompt_tokens
                        scope["total_cached_tokens"] += cached_tokens
                        scope["total_completion_tokens"] += completion_tokens
                        scope["total_tokens"] += total_tokens
                        scope["api_call_count"] += 1

                return response.choices[0].message.content
            except (APITimeoutError, asyncio.TimeoutError) as exc:
                last_exc = exc
                timeout_escalations += 1
                if attempt == self.max_retries:
                    break
                wait = min(2 ** attempt, 60)
                logging.warning(
                    f"[LLM Retry] Attempt {attempt}/{self.max_retries} timed out after "
                    f"{self.timeout}s ({type(exc).__name__}). Escalating temperature to "
                    f"{min(base_temperature + timeout_escalations * self.temp_step, 1.0):.2f}. "
                    f"Retrying in {wait:.2f}s..."
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                wait = min(2 ** attempt, 60)
                logging.warning(
                    f"[LLM Retry] Attempt {attempt}/{self.max_retries} failed with "
                    f"{type(exc).__name__}: {exc}. Retrying in {wait:.2f}s..."
                )
                await asyncio.sleep(wait)

        raise last_exc

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Call the LLM with system and user prompts.

        Args:
            system_prompt: System-level instructions
            user_prompt: User query
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            LLM response content
        """
        return await self._completion_request(system_prompt, user_prompt, temperature)

    def get_token_stats(self) -> dict:
        """
        Get current token usage statistics.

        Returns:
            Dictionary with token counts and API call count
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "api_call_count": self.api_call_count,
        }

    @staticmethod
    def _usage_value(usage: Any, field: str) -> int:
        """Read one token count from an SDK object or mapping."""
        if usage is None:
            return 0
        if isinstance(usage, dict):
            value = usage.get(field, 0)
        else:
            value = getattr(usage, field, 0)
        return int(value or 0)

    @classmethod
    def _cached_token_count(cls, usage: Any) -> int:
        """Read cache-read tokens from OpenAI-compatible usage objects.

        Chat Completions reports this as
        ``prompt_tokens_details.cached_tokens``.  The additional forms keep
        tracking compatible with Responses-style and provider-specific
        OpenAI-compatible responses.
        """
        for details_field in ("prompt_tokens_details", "input_tokens_details"):
            if isinstance(usage, dict):
                details = usage.get(details_field)
            else:
                details = getattr(usage, details_field, None)
            if details is not None:
                cached_tokens = cls._usage_value(details, "cached_tokens")
                if cached_tokens:
                    return cached_tokens

        for field in ("cached_tokens", "cached_input_tokens", "cache_read_input_tokens"):
            cached_tokens = cls._usage_value(usage, field)
            if cached_tokens:
                return cached_tokens
        return 0

    @staticmethod
    def _empty_token_stats() -> Dict[str, int]:
        return {
            "total_prompt_tokens": 0,
            "total_cached_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "api_call_count": 0,
        }

    def start_token_scope(self) -> Token:
        """Start an asyncio-task-local token counter for one dataset sample."""
        return self._token_scope.set(self._empty_token_stats())

    def get_token_scope_stats(self) -> Optional[Dict[str, int]]:
        """Return a snapshot of the active sample's token counter, if any."""
        stats = self._token_scope.get()
        return dict(stats) if stats is not None else None

    def end_token_scope(self, token: Token) -> Dict[str, int]:
        """Finish the active sample counter and restore the previous context."""
        stats = self.get_token_scope_stats() or self._empty_token_stats()
        self._token_scope.reset(token)
        return stats

    def reset_token_stats(self):
        """Reset all token usage statistics."""
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
