"""
Abstract base class for all LLM clients.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass


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


class BaseClient(ABC):
    """Abstract base class for all LLM clients."""

    def __init__(self, model: str, name: str, **kwargs):
        """
        Initialize the client.

        Args:
            model: Model identifier
            name: Display name for the client
            **kwargs: Provider-specific parameters
        """
        self.model = model
        self.name = name
        self.messages: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.stream_callback: Optional[Callable] = None

    @abstractmethod
    def ask(self, question: str, round_num: int = 0) -> Tuple[str, Dict]:
        """
        Send a question to the LLM and get response with token counts.

        Args:
            question: The question/prompt to send
            round_num: Current round number (for debugging/logging)

        Returns:
            Tuple of (response_text, token_info_dict)

        Raises:
            Exception: If the request fails
        """
        pass

    @abstractmethod
    def check_server_available(self) -> bool:
        """
        Check if the service is available.

        Returns:
            True if service is available, False otherwise
        """
        pass

    @abstractmethod
    def reset_conversation(self) -> None:
        """Reset conversation history and token counts."""
        pass

    def set_stream_callback(self, callback: Callable) -> None:
        """
        Set streaming callback (optional for non-streaming APIs).

        Args:
            callback: Function to call with streaming chunks
        """
        self.stream_callback = callback

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role (e.g., 'system', 'user', 'assistant')
            content: Message content
        """
        self.messages.append({'role': role, 'content': content})

    def get_conversation_length(self) -> int:
        """
        Get current conversation length.

        Returns:
            Number of messages in conversation
        """
        return len(self.messages)

    def get_total_tokens(self) -> int:
        """
        Get total tokens used across all requests.

        Returns:
            Total token count
        """
        return self.total_tokens
