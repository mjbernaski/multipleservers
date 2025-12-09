"""
OpenAI client implementation.
"""
import os
import time
from typing import Dict, Tuple, Optional
from .base_client import BaseClient

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIClient(BaseClient):
    """Client for communicating with OpenAI's API."""

    MODELS = {
        # GPT-4o series (latest)
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        # GPT-4 Turbo
        'gpt-4-turbo': 'gpt-4-turbo',
        # GPT-4
        'gpt-4': 'gpt-4',
        # GPT-3.5
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        # o1 reasoning models
        'o1': 'o1',
        'o1-mini': 'o1-mini',
        'o1-preview': 'o1-preview',
    }

    # Models that support reasoning/thinking
    REASONING_MODELS = {'o1', 'o1-mini', 'o1-preview'}

    def __init__(self, model: str, name: str = None, api_key: str = None,
                 temperature: float = None, max_tokens: int = 4096,
                 reasoning_effort: str = None, role: str = None):
        """
        Initialize OpenAI client.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'o1', 'o1-mini')
            name: Display name for the client
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Sampling temperature (0.0-2.0, not for reasoning models)
            max_tokens: Maximum tokens in response
            reasoning_effort: For o1/o3 models: 'low', 'medium', 'high'
            role: Role in dialog (intermediator, participant1, participant2)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        actual_model = self.MODELS.get(model, model)
        super().__init__(model=actual_model, name=name or f"OpenAI ({model})", host="api.openai.com")

        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment or provided")

        self.client = OpenAI(api_key=self.api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.role = role
        self.model_alias = model
        self.provider = 'openai'

        # Check if this is a reasoning model
        self.is_reasoning_model = model in self.REASONING_MODELS or actual_model in self.REASONING_MODELS
        self.thinking = self.is_reasoning_model

        # System messages storage
        self.system_messages: list = []

        # Thinking content storage (for o1/o3 reasoning)
        self.thinking_history: list = []
        self.last_thinking: str = ""

        # Separate token tracking
        self.total_thinking_tokens = 0
        self.total_speaking_tokens = 0

    def check_server_available(self) -> bool:
        """Check if the OpenAI API is accessible."""
        try:
            # List models to verify API access
            self.client.models.list()
            print(f"✓ Connected to OpenAI API")
            print(f"✓ Model '{self.model_alias}' selected")
            return True
        except Exception as e:
            error_str = str(e)
            if "authentication" in error_str.lower() or "api key" in error_str.lower():
                print(f"Error: Invalid OpenAI API key")
            else:
                print(f"Error: Failed to connect to OpenAI API: {e}")
            return False

    def ask(self, question: str, round_num: int = 0, phase: str = None) -> Tuple[str, Dict]:
        """Send a question to OpenAI and get response with token counts."""
        self.messages.append({"role": "user", "content": question})

        # Extract system prompt from messages if set there (dialog system compatibility)
        system_msgs = self.system_messages
        if not system_msgs:
            for msg in self.messages:
                if msg.get("role") in ("system", "developer"):
                    role = "developer" if self.is_reasoning_model else "system"
                    system_msgs = [{"role": role, "content": msg.get("content", "")}]
                    break

        # Build messages list with system messages first
        api_messages = system_msgs + self._get_api_messages()

        # Build request parameters
        params = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
        }

        # Add max_tokens (different param name for reasoning models)
        if self.is_reasoning_model:
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["max_tokens"] = self.max_tokens

        # Add temperature (not supported for reasoning models)
        if self.temperature is not None and not self.is_reasoning_model:
            params["temperature"] = self.temperature

        # Add reasoning effort for o1/o3 models
        if self.reasoning_effort and self.is_reasoning_model:
            params["reasoning_effort"] = self.reasoning_effort

        try:
            start_time = time.time()
            first_token_time = None
            answer = ""
            thinking_content = ""
            prompt_tokens = 0
            completion_tokens = 0
            reasoning_tokens = 0

            # Stream the response
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Handle content
                    if delta.content:
                        text = delta.content
                        answer += text
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                        if self.stream_callback:
                            self.stream_callback({
                                'type': 'content',
                                'content': text,
                                'name': self.name
                            })

                # Get usage from final chunk
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    if hasattr(chunk.usage, 'completion_tokens_details'):
                        details = chunk.usage.completion_tokens_details
                        if details and hasattr(details, 'reasoning_tokens'):
                            reasoning_tokens = details.reasoning_tokens or 0

            total = prompt_tokens + completion_tokens
            ttft = first_token_time if first_token_time else 0
            elapsed = time.time() - start_time
            tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0

            # For reasoning models, reasoning tokens represent "thinking"
            if self.is_reasoning_model and reasoning_tokens > 0:
                thinking_content = f"[Reasoning: {reasoning_tokens} tokens used]"
                self.last_thinking = thinking_content
                self.thinking_history.append({
                    'round': round_num,
                    'phase': phase,
                    'thinking': thinking_content,
                    'reasoning_tokens': reasoning_tokens
                })

            # Store message
            message_entry = {"role": "assistant", "content": answer}
            if thinking_content:
                message_entry["thinking"] = thinking_content
            self.messages.append(message_entry)

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total

            # Token split for reasoning models
            thinking_tokens = reasoning_tokens
            speaking_tokens = completion_tokens - reasoning_tokens

            self.total_thinking_tokens += thinking_tokens
            self.total_speaking_tokens += speaking_tokens

            token_info = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'thinking_tokens': thinking_tokens,
                'speaking_tokens': speaking_tokens,
                'total': total,
                'tokens_per_second': tokens_per_second,
                'time_to_first_token': ttft,
                'thinking': thinking_content if thinking_content else None,
                'reasoning_tokens': reasoning_tokens
            }

            if self.stream_callback:
                self.stream_callback({
                    'type': 'response_complete',
                    'answer': answer,
                    'tokens': token_info,
                    'thinking': thinking_content if thinking_content else None,
                    'name': self.name
                })

            return answer, token_info

        except Exception as e:
            error_msg = f"OpenAI API error: {e}"
            if self.stream_callback:
                self.stream_callback({
                    'type': 'error',
                    'error': error_msg,
                    'name': self.name
                })
            raise Exception(error_msg)

    def _get_api_messages(self) -> list:
        """Get messages formatted for the API.

        OpenAI works best with alternating user/assistant messages.
        This method filters out system messages (handled separately)
        and merges consecutive messages from the same role.
        """
        # Filter out system messages (they're handled separately)
        filtered = [msg for msg in self.messages if msg["role"] not in ("system", "developer")]

        if not filtered:
            return []

        # Merge consecutive messages with the same role
        merged = []
        for msg in filtered:
            if merged and merged[-1]["role"] == msg["role"]:
                # Merge with previous message
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append({"role": msg["role"], "content": msg["content"]})

        return merged

    def reset_conversation(self):
        """Reset conversation history and token counts."""
        self.messages = []
        self.system_messages = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_thinking_tokens = 0
        self.total_speaking_tokens = 0
        self.thinking_history = []
        self.last_thinking = ""

    def set_system_prompt(self, prompt: str):
        """Set the system prompt for conversations.

        Also clears the messages list to ensure a fresh conversation.
        """
        # For reasoning models, use developer role; otherwise system
        role = "developer" if self.is_reasoning_model else "system"
        self.system_messages = [{"role": role, "content": prompt}]
        self.messages = []

    def get_full_context(self) -> dict:
        """Get the full context window contents and metadata."""
        total_chars = sum(len(msg.get('content', '')) for msg in self.messages)
        thinking_chars = sum(len(msg.get('thinking', '')) for msg in self.messages if 'thinking' in msg)

        return {
            'name': self.name,
            'model': self.model,
            'provider': 'openai',
            'role': self.role,
            'thinking_enabled': self.is_reasoning_model,
            'reasoning_effort': self.reasoning_effort,
            'messages': self.messages,
            'thinking_history': self.thinking_history,
            'last_thinking': self.last_thinking,
            'stats': {
                'message_count': len(self.messages),
                'total_chars': total_chars,
                'thinking_chars': thinking_chars,
                'estimated_tokens': total_chars // 4,
                'total_prompt_tokens': self.total_prompt_tokens,
                'total_completion_tokens': self.total_completion_tokens,
                'total_thinking_tokens': self.total_thinking_tokens,
                'total_speaking_tokens': self.total_speaking_tokens,
                'total_tokens': self.total_tokens,
            },
            'parameters': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'reasoning_effort': self.reasoning_effort
            }
        }
