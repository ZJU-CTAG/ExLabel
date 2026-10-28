import aiohttp
import asyncio
import numpy as np
import logging
import os
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential, 
    retry_if_exception_type, 
    RetryCallState
)


def log_retry(retry_state: RetryCallState):
    """Log retry attempts for debugging purposes."""
    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    logging.warning(
        f"[Embedder Retry] Attempt {attempt} failed with {type(exc).__name__}: {exc}. "
        f"Retrying in {wait:.2f}s..."
    )


class Embedder:
    """
    Async text embedding client with automatic retry logic.
    
    Provides asynchronous interface to embedding models with:
    - Automatic retry on failures
    - Configurable timeout
    - Error handling for rate limits
    """
    
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = None,
        max_retries: int = 3,
        timeout: int = 180
    ):
        """
        Initialize the embedder client.
        
        Args:
            model: Embedding model name
            base_url: API URL endpoint
            api_key: API key for authentication (defaults to EMBEDDER_API_KEY from env)
            max_retries: Maximum attempts (three by default)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("EMBEDDER_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key must be provided or set EMBEDDER_API_KEY environment variable")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = max_retries
        self.timeout = timeout
        self._lock = asyncio.Lock()

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
        before_sleep=log_retry
    )
    async def _embed_request(self, text: str) -> np.ndarray:
        """
        Make an embedding request to the API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            async with session.post(
                self.base_url, 
                json=payload, 
                headers=self.headers
            ) as response:
                if response.status == 429:
                    raise aiohttp.ClientError("Rate limit hit (429)")
                response.raise_for_status()
                data = await response.json()

        embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
        return embedding

    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return await self._embed_request(text)
