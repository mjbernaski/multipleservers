"""
Google Gemini client implementation.
"""
import os
import time
from typing import Dict, Tuple, Optional
from .base_client import BaseClient, ClientConnectionError, ClientAuthError, ClientRateLimitError, ClientTimeoutError, ClientError

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiClient(BaseClient):
    """Client for communicating with Google's Gemini API."""

    MODELS = {
        # Gemini 3.0 series (preview)
        'gemini-3-pro': 'gemini-3-pro-preview',
        # Gemini 2.5 series
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
        # Gemini 2.0 series
        'gemini-2-flash': 'gemini-2.0-flash',
        'gemini-2-flash-lite': 'gemini-2.0-flash-lite',
        'gemini-2-pro-exp': 'gemini-2.0-pro-exp',
    }

    # Models that support deep thinking
    THINKING_MODELS = {'gemini-3-deep-think', 'gemini-3.0-deep-think', 'gemini-2.5-pro', 'gemini-2.5-flash'}

    def __init__(self, model: str, name: str = None, api_key: str = None,
                 temperature: float = None, max_tokens: int = 4096,
                 thinking_level: str = None, role: str = None):
        """
        Initialize Gemini client.

        Args:
            model: Model identifier (e.g., 'gemini-3-pro', 'gemini-2.5-flash')
            name: Display name for the client
            api_key: Google AI API key (defaults to GOOGLE_API_KEY env var)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            thinking_level: For thinking models: 'low', 'medium', 'high'
            role: Role in dialog (intermediator, participant1, participant2)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        actual_model = self.MODELS.get(model, model)
        super().__init__(model=actual_model, name=name or f"Gemini ({model})", host="generativelanguage.googleapis.com")

        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment or provided")

        genai.configure(api_key=self.api_key)
        self.model_alias = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level
        self.role = role
        self.provider = 'gemini'

        # Check if this is a thinking model
        self.is_thinking_model = model in self.THINKING_MODELS or actual_model in self.THINKING_MODELS
        self.thinking = self.is_thinking_model and thinking_level is not None

        # System instruction storage
        self.system_instruction: Optional[str] = None

        # Thinking content storage
        self.thinking_history: list = []
        self.last_thinking: str = ""

        # Separate token tracking
        self.total_thinking_tokens = 0
        self.total_speaking_tokens = 0

        # Initialize the model
        self._init_model()

    def _init_model(self):
        """Initialize or reinitialize the Gemini model."""
        generation_config = {
            "max_output_tokens": self.max_tokens,
        }

        if self.temperature is not None:
            generation_config["temperature"] = self.temperature

        # Add thinking level for supported models
        if self.thinking_level and self.is_thinking_model:
            generation_config["thinking_config"] = {"thinking_level": self.thinking_level}

        self.genai_model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            system_instruction=self.system_instruction
        )

        # Start a chat session
        self.chat = self.genai_model.start_chat(history=[])

    def check_server_available(self) -> bool:
        """Check if the Gemini API is accessible."""
        try:
            # List models to verify API access
            models = genai.list_models()
            model_names = [m.name for m in models]
            print(f"✓ Connected to Google Gemini API")
            print(f"✓ Model '{self.model_alias}' selected")
            return True
        except Exception as e:
            error_str = str(e)
            if "api key" in error_str.lower() or "authentication" in error_str.lower():
                print(f"Error: Invalid Google API key")
            else:
                print(f"Error: Failed to connect to Gemini API: {e}")
            return False

    def ask(self, question: str, round_num: int = 0, phase: str = None) -> Tuple[str, Dict]:
        """Send a question to Gemini and get response with token counts."""
        # Extract system prompt from messages if set there (dialog system compatibility)
        if not self.system_instruction:
            for msg in self.messages:
                if msg.get("role") == "system":
                    self.system_instruction = msg.get("content", "")
                    self._init_model()  # Reinitialize with system instruction
                    break

        # NOTE: The Gemini chat session maintains its own history which can get
        # out of sync with self.messages. self.messages is the source of truth.
        self.messages.append({"role": "user", "content": question})

        try:
            start_time = time.time()
            first_token_time = None
            answer = ""
            thinking_content = ""

            # Stream the response (with retry for transient errors)
            retryable = (ConnectionError, TimeoutError)
            if GEMINI_AVAILABLE:
                from google.api_core import exceptions as google_exceptions
                retryable = (ConnectionError, TimeoutError, google_exceptions.ServiceUnavailable,
                             google_exceptions.InternalServerError, google_exceptions.TooManyRequests)
            response = self._retry_with_backoff(
                lambda: self.chat.send_message(question, stream=True),
                retryable
            )

            for chunk in response:
                if chunk.text:
                    text = chunk.text
                    answer += text
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    if self.stream_callback:
                        self.stream_callback({
                            'type': 'content',
                            'content': text,
                            'name': self.name
                        })

                # Check for thinking content in candidates
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'thinking_content') and candidate.thinking_content:
                            thinking_content += candidate.thinking_content
                            if self.stream_callback:
                                self.stream_callback({
                                    'type': 'thinking',
                                    'content': candidate.thinking_content,
                                    'name': self.name
                                })

            # Get token counts from usage metadata
            prompt_tokens = 0
            completion_tokens = 0
            thinking_tokens = 0

            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
                completion_tokens = getattr(usage, 'candidates_token_count', 0)
                thinking_tokens = getattr(usage, 'thinking_token_count', 0)

            total = prompt_tokens + completion_tokens
            ttft = first_token_time if first_token_time else 0
            elapsed = time.time() - start_time
            tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0

            # Store thinking content
            if thinking_content:
                self.last_thinking = thinking_content
                self.thinking_history.append({
                    'round': round_num,
                    'phase': phase,
                    'thinking': thinking_content,
                    'thinking_tokens': thinking_tokens
                })
            elif thinking_tokens > 0:
                # Model used thinking but didn't expose content
                self.last_thinking = f"[Thinking: {thinking_tokens} tokens used]"
                self.thinking_history.append({
                    'round': round_num,
                    'phase': phase,
                    'thinking': self.last_thinking,
                    'thinking_tokens': thinking_tokens
                })

            # Store message
            message_entry = {"role": "assistant", "content": answer}
            if thinking_content or thinking_tokens > 0:
                message_entry["thinking"] = thinking_content or self.last_thinking
            self.messages.append(message_entry)

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total

            # Token split
            speaking_tokens = completion_tokens - thinking_tokens if completion_tokens > thinking_tokens else completion_tokens

            self.total_thinking_tokens += thinking_tokens
            self.total_speaking_tokens += speaking_tokens

            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            token_info = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'thinking_tokens': thinking_tokens,
                'speaking_tokens': speaking_tokens,
                'total': total,
                'tokens_per_second': tokens_per_second,
                'time_to_first_token': ttft,
                'thinking': thinking_content if thinking_content else (self.last_thinking if thinking_tokens > 0 else None),
                'cost': cost.to_dict()
            }

            if self.stream_callback:
                self.stream_callback({
                    'type': 'response_complete',
                    'answer': answer,
                    'tokens': token_info,
                    'thinking': token_info['thinking'],
                    'name': self.name
                })

            return answer, token_info

        except (ConnectionError, OSError) as e:
            error_msg = f"Gemini connection error: {e}"
            if self.stream_callback:
                self.stream_callback({
                    'type': 'error',
                    'error': error_msg,
                    'name': self.name
                })
            raise ClientConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Gemini API error: {e}"
            if self.stream_callback:
                self.stream_callback({
                    'type': 'error',
                    'error': error_msg,
                    'name': self.name
                })
            raise ClientError(error_msg) from e

    def reset_conversation(self):
        """Reset conversation history and token counts."""
        self.messages = []
        self.system_instruction = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_thinking_tokens = 0
        self.total_speaking_tokens = 0
        self.thinking_history = []
        self.last_thinking = ""
        self._init_model()

    def set_system_prompt(self, prompt: str):
        """Set the system instruction for conversations.

        Also clears the messages list to ensure a fresh conversation.
        """
        self.system_instruction = prompt
        self.messages = []
        # Reinitialize model with new system instruction (also resets chat session)
        self._init_model()

    def get_full_context(self) -> dict:
        """Get the full context window contents and metadata."""
        total_chars = sum(len(msg.get('content', '')) for msg in self.messages)
        thinking_chars = sum(len(msg.get('thinking', '')) for msg in self.messages if 'thinking' in msg)

        return {
            'name': self.name,
            'model': self.model,
            'provider': 'gemini',
            'role': self.role,
            'thinking_enabled': self.thinking,
            'thinking_level': self.thinking_level,
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
                'thinking_level': self.thinking_level
            }
        }
