"""
Abstract base class for all LLM clients.
"""
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, field


# =========================================================================
# ERROR HIERARCHY
# =========================================================================

class ClientError(Exception):
    """Base error for all client operations."""
    pass


class ClientConnectionError(ClientError):
    """Server/API unreachable."""
    pass


class ClientAuthError(ClientError):
    """Authentication/API key error."""
    pass


class ClientRateLimitError(ClientError):
    """Rate limit exceeded."""
    pass


class ClientTimeoutError(ClientError):
    """Request timed out."""
    pass


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass
class RetryConfig:
    """Configuration for retry with exponential backoff."""
    max_retries: int = 2
    base_delay: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class TokenInfo:
    """Token usage information for a response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: Optional[float] = None
    time_to_first_token: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'tokens_per_second': self.tokens_per_second,
            'time_to_first_token': self.time_to_first_token
        }


@dataclass
class ThinkingInfo:
    """Thinking/reasoning token information."""
    thinking_tokens: int = 0
    speaking_tokens: int = 0
    is_estimated: bool = True

    def to_dict(self) -> Dict:
        return {
            'thinking_tokens': self.thinking_tokens,
            'speaking_tokens': self.speaking_tokens,
            'is_estimated': self.is_estimated
        }


@dataclass
class CostInfo:
    """Per-request cost tracking."""
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    def to_dict(self) -> Dict:
        return {
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost,
            'currency': self.currency
        }


# Per-provider pricing (per 1M tokens)
PROVIDER_PRICING = {
    'ollama': {'input': 0.0, 'output': 0.0},
    'anthropic': {
        'claude-opus-4-5-20251101': {'input': 15.0, 'output': 75.0},
        'claude-sonnet-4-5-20250929': {'input': 3.0, 'output': 15.0},
        'claude-haiku-4-5-20251001': {'input': 0.80, 'output': 4.0},
        'claude-opus-4-1-20250805': {'input': 15.0, 'output': 75.0},
        'claude-opus-4-20250514': {'input': 15.0, 'output': 75.0},
        'claude-sonnet-4-20250514': {'input': 3.0, 'output': 15.0},
        '_default': {'input': 3.0, 'output': 15.0},
    },
    'openai': {
        'gpt-4o': {'input': 2.50, 'output': 10.0},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-4.1': {'input': 2.0, 'output': 8.0},
        'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
        'o1': {'input': 15.0, 'output': 60.0},
        'o1-mini': {'input': 1.10, 'output': 4.40},
        'o3-mini': {'input': 1.10, 'output': 4.40},
        '_default': {'input': 2.50, 'output': 10.0},
    },
    'gemini': {
        'gemini-2.5-pro': {'input': 1.25, 'output': 10.0},
        'gemini-2.5-flash': {'input': 0.15, 'output': 0.60},
        '_default': {'input': 1.25, 'output': 10.0},
    },
}


# =========================================================================
# BASE CLIENT
# =========================================================================

class BaseClient(ABC):
    """Abstract base class for all LLM clients."""

    def __init__(self, model: str, name: str, host: str = None,
                 timeout: int = 300, retry_config: RetryConfig = None, **kwargs):
        self.model = model
        self.name = name
        self.host = host or "cloud"
        self.provider = 'unknown'
        self.messages: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.stream_callback: Optional[Callable] = None
        self.thinking = False
        self.role: Optional[str] = None
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()

    def _retry_with_backoff(self, func, retryable_exceptions, *args, **kwargs):
        """Execute func with retry and exponential backoff.

        Args:
            func: Callable to execute
            retryable_exceptions: Tuple of exception types to retry on
            *args, **kwargs: Passed to func

        Returns:
            Result of func

        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt)
                    if self.stream_callback:
                        self.stream_callback({
                            'type': 'retry',
                            'attempt': attempt + 1,
                            'max_retries': self.retry_config.max_retries,
                            'delay': delay,
                            'error': str(e),
                            'name': self.name
                        })
                    print(f"[{self.name}] Retry {attempt + 1}/{self.retry_config.max_retries} after {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    raise

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> CostInfo:
        """Calculate cost for a request based on provider pricing."""
        provider_prices = PROVIDER_PRICING.get(self.provider, {})

        if isinstance(provider_prices, dict) and 'input' in provider_prices:
            prices = provider_prices
        else:
            prices = provider_prices.get(self.model, provider_prices.get('_default', {'input': 0, 'output': 0}))

        input_cost = (prompt_tokens / 1_000_000) * prices.get('input', 0)
        output_cost = (completion_tokens / 1_000_000) * prices.get('output', 0)
        total = input_cost + output_cost

        self.total_cost += total

        return CostInfo(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total
        )

    @abstractmethod
    def ask(self, question: str, round_num: int = 0, phase: str = None) -> Tuple[str, Dict]:
        pass

    @abstractmethod
    def check_server_available(self) -> bool:
        pass

    @abstractmethod
    def reset_conversation(self) -> None:
        pass

    def set_stream_callback(self, callback: Callable) -> None:
        self.stream_callback = callback

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({'role': role, 'content': content})

    def get_conversation_length(self) -> int:
        return len(self.messages)

    def get_total_tokens(self) -> int:
        return self.total_tokens

    def preload_model(self) -> bool:
        return True

    def get_full_context(self) -> dict:
        total_chars = sum(len(msg.get('content', '')) for msg in self.messages)
        return {
            'name': self.name,
            'model': self.model,
            'provider': self.provider,
            'role': self.role,
            'thinking_enabled': self.thinking,
            'messages': self.messages,
            'stats': {
                'message_count': len(self.messages),
                'total_chars': total_chars,
                'estimated_tokens': total_chars // 4,
                'total_prompt_tokens': self.total_prompt_tokens,
                'total_completion_tokens': self.total_completion_tokens,
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
            }
        }
