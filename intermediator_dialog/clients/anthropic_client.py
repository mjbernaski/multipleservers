"""
Anthropic Claude client implementation.
"""
import os
import time
from typing import Dict, Tuple, Optional
from .base_client import BaseClient

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicClient(BaseClient):
    """Client for communicating with Anthropic's Claude API."""

    MODELS = {
        # Claude 4 series (2025)
        'claude-sonnet-4': 'claude-sonnet-4-20250514',
        'claude-opus-4': 'claude-opus-4-20250514',
        # Claude 3.5 series
        'claude-sonnet-3.5': 'claude-3-5-sonnet-20241022',
        'claude-haiku-3.5': 'claude-3-5-haiku-20241022',
        # Claude 3 series (legacy)
        'claude-opus-3': 'claude-3-opus-20240229',
        'claude-sonnet-3': 'claude-3-sonnet-20240229',
        'claude-haiku-3': 'claude-3-haiku-20240307',
    }

    def __init__(self, model: str, name: str = None, api_key: str = None,
                 temperature: float = None, max_tokens: int = 4096,
                 thinking: bool = False, thinking_budget: int = 10000,
                 role: str = None):
        """
        Initialize Anthropic client.

        Args:
            model: Model identifier (e.g., 'claude-sonnet-4', 'claude-haiku-3.5')
            name: Display name for the client
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            thinking: Enable extended thinking (for supported models)
            thinking_budget: Token budget for thinking (when enabled)
            role: Role in dialog (intermediator, participant1, participant2)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        # Resolve model alias to actual model ID
        actual_model = self.MODELS.get(model, model)
        super().__init__(model=actual_model, name=name or f"Claude ({model})", host="api.anthropic.com")

        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment or provided")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.role = role
        self.model_alias = model
        self.provider = 'anthropic'

        # System prompt storage
        self.system_prompt: Optional[str] = None

        # Thinking content storage
        self.thinking_history: list = []
        self.last_thinking: str = ""

        # Separate token tracking
        self.total_thinking_tokens = 0
        self.total_speaking_tokens = 0

    def check_server_available(self) -> bool:
        """Check if the Anthropic API is accessible."""
        print(f"[Anthropic] Checking availability for model: {self.model}")
        print(f"[Anthropic] API key configured: {'Yes' if self.api_key else 'No'}")
        try:
            # Make a minimal request to verify API access
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print(f"✓ Connected to Anthropic API")
            print(f"✓ Model '{self.model_alias}' ({self.model}) is available")
            return True
        except anthropic.AuthenticationError as e:
            print(f"Error: Invalid Anthropic API key: {e}")
            return False
        except anthropic.NotFoundError as e:
            print(f"Error: Model '{self.model}' not found: {e}")
            return False
        except Exception as e:
            print(f"Error: Failed to connect to Anthropic API: {type(e).__name__}: {e}")
            return False

    def ask(self, question: str, round_num: int = 0, phase: str = None) -> Tuple[str, Dict]:
        """Send a question to Claude and get response with token counts."""
        self.messages.append({"role": "user", "content": question})

        # Extract system prompt from messages if set there (dialog system compatibility)
        system_prompt = self.system_prompt
        if not system_prompt:
            for msg in self.messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    break

        # Build request parameters
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._get_api_messages(),
        }

        # Add system prompt if set
        if system_prompt:
            params["system"] = system_prompt

        # Add temperature if set (not compatible with extended thinking)
        if self.temperature is not None and not self.thinking:
            params["temperature"] = self.temperature

        # Enable extended thinking if requested
        if self.thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }

        try:
            start_time = time.time()
            first_token_time = None
            answer = ""
            thinking_content = ""

            # Use streaming
            with self.client.messages.stream(**params) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                if event.content_block.type == 'thinking':
                                    pass  # Thinking block starting
                                elif event.content_block.type == 'text':
                                    pass  # Text block starting

                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                if hasattr(event.delta, 'thinking'):
                                    thinking_content += event.delta.thinking
                                    if self.stream_callback:
                                        self.stream_callback({
                                            'type': 'thinking',
                                            'content': event.delta.thinking,
                                            'name': self.name
                                        })
                                elif hasattr(event.delta, 'text'):
                                    text = event.delta.text
                                    answer += text
                                    if first_token_time is None and text:
                                        first_token_time = time.time() - start_time
                                    if self.stream_callback:
                                        self.stream_callback({
                                            'type': 'content',
                                            'content': text,
                                            'name': self.name
                                        })

                # Get final message for token counts
                final_message = stream.get_final_message()

            prompt_tokens = final_message.usage.input_tokens
            completion_tokens = final_message.usage.output_tokens
            total = prompt_tokens + completion_tokens

            ttft = first_token_time if first_token_time else 0
            elapsed = time.time() - start_time
            tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0

            # Store thinking content
            self.last_thinking = thinking_content
            if thinking_content:
                self.thinking_history.append({
                    'round': round_num,
                    'phase': phase,
                    'thinking': thinking_content
                })

            # Store message
            message_entry = {"role": "assistant", "content": answer}
            if thinking_content:
                message_entry["thinking"] = thinking_content
            self.messages.append(message_entry)

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total

            # Estimate thinking vs speaking token split
            thinking_tokens = 0
            speaking_tokens = completion_tokens
            if thinking_content and answer:
                total_chars = len(thinking_content) + len(answer)
                if total_chars > 0:
                    thinking_ratio = len(thinking_content) / total_chars
                    thinking_tokens = int(completion_tokens * thinking_ratio)
                    speaking_tokens = completion_tokens - thinking_tokens
            elif thinking_content and not answer:
                thinking_tokens = completion_tokens
                speaking_tokens = 0

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
                'thinking': thinking_content if thinking_content else None
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

        except anthropic.APIError as e:
            error_msg = f"Anthropic API error: {e}"
            print(f"[AnthropicClient] Error in ask(): {error_msg}")
            print(f"[AnthropicClient] Message count: {len(self.messages)}, API message count: {len(self._get_api_messages())}")
            if self.stream_callback:
                self.stream_callback({
                    'type': 'error',
                    'error': error_msg,
                    'name': self.name
                })
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Anthropic error: {type(e).__name__}: {e}"
            print(f"[AnthropicClient] Unexpected error in ask(): {error_msg}")
            print(f"[AnthropicClient] Message count: {len(self.messages)}, API message count: {len(self._get_api_messages())}")
            if self.stream_callback:
                self.stream_callback({
                    'type': 'error',
                    'error': error_msg,
                    'name': self.name
                })
            raise Exception(error_msg)

    def _get_api_messages(self) -> list:
        """Get messages formatted for the API (excluding system prompt).

        Anthropic API requires alternating user/assistant messages.
        This method merges consecutive messages from the same role.
        """
        # Filter out system messages first
        filtered = [msg for msg in self.messages if msg["role"] != "system"]

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
        self.system_prompt = None
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
        self.system_prompt = prompt
        self.messages = []

    def get_full_context(self) -> dict:
        """Get the full context window contents and metadata."""
        total_chars = sum(len(msg.get('content', '')) for msg in self.messages)
        thinking_chars = sum(len(msg.get('thinking', '')) for msg in self.messages if 'thinking' in msg)

        return {
            'name': self.name,
            'model': self.model,
            'provider': 'anthropic',
            'role': self.role,
            'thinking_enabled': self.thinking,
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
                'thinking_budget': self.thinking_budget if self.thinking else None
            }
        }
