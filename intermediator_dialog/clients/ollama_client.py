"""
Ollama client implementation.
"""
import requests
import json
import time
from typing import Dict, Tuple
from .base_client import BaseClient


class OllamaClient(BaseClient):
    """Client for communicating with Ollama servers."""

    def __init__(self, host: str, model: str, name: str = None, num_ctx: int = 96000,
                 temperature: float = None, top_p: float = None, top_k: int = None,
                 repeat_penalty: float = None, num_predict: int = None, thinking: bool = False,
                 reasoning_effort: str = None, be_brief: bool = False, keep_alive: str = "10m",
                 role: str = None):
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL
            model: Model identifier
            name: Display name for the client
            num_ctx: Context window size
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            num_predict: Maximum tokens to predict
            thinking: Enable thinking mode (for DeepSeek-R1 style models)
            reasoning_effort: Reasoning effort level for GPT-OSS models ("low", "medium", "high")
            be_brief: Prepend "Be Brief. " to all prompts
            keep_alive: Model keep-alive duration
            role: Role in dialog (intermediator, participant1, participant2)
        """
        super().__init__(model=model, name=name or f"{host} ({model})")

        self.host = host.rstrip('/')
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.num_predict = num_predict
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.be_brief = be_brief
        self.keep_alive = keep_alive
        self.role = role

        # Thinking content storage - stores thinking from each response
        self.thinking_history: list = []
        self.last_thinking: str = ""

        # Separate token tracking for thinking vs speaking
        self.total_thinking_tokens = 0
        self.total_speaking_tokens = 0

    def preload_model(self) -> bool:
        """Preload the model into memory by sending a minimal request."""
        try:
            url = f"{self.host}/api/generate"

            payload = {
                "model": self.model,
                "prompt": "Hi",
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_predict": 1  # Only generate 1 token to minimize time
                }
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            print(f"✓ Preloaded model '{self.model}' on {self.host}")
            return True

        except Exception as e:
            print(f"Warning: Failed to preload model '{self.model}' on {self.host}: {e}")
            return False

    def check_server_available(self) -> bool:
        """Check if the Ollama server is available and the model exists."""
        try:
            url = f"{self.host}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            models = response.json().get('models', [])
            model_names = [m.get('name') for m in models]

            # If no model is specified, just check server connectivity
            if not self.model:
                print(f"✓ Connected to Ollama server at {self.host}")
                return True

            if self.model not in model_names:
                print(f"Error: Model '{self.model}' not found on server {self.host}")
                print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                return False

            print(f"✓ Connected to Ollama server at {self.host}")
            print(f"✓ Model '{self.model}' is available")
            return True

        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Ollama server at {self.host}")
            return False
        except requests.exceptions.Timeout:
            print(f"Error: Connection to {self.host} timed out")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to communicate with Ollama server: {e}")
            return False

    def ask(self, question: str, round_num: int = 0, phase: str = None) -> Tuple[str, Dict]:
        """Send a question to Ollama and get response with token counts."""
        url = f"{self.host}/api/chat"

        if self.be_brief:
            question = "Be Brief. " + question

        self.messages.append({"role": "user", "content": question})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "keep_alive": self.keep_alive
        }

        if self.thinking:
            payload["thinking"] = True

        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        if self.num_ctx is not None:
            payload["num_ctx"] = self.num_ctx
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.repeat_penalty is not None:
            payload["repeat_penalty"] = self.repeat_penalty
        if self.num_predict is not None:
            payload["num_predict"] = self.num_predict

        try:
            response = requests.post(url, json=payload, timeout=300, stream=True)

            if response.status_code >= 400:
                try:
                    # Try to read error message from response
                    error_content = response.text
                    print(f"Ollama API Error ({response.status_code}): {error_content}")
                except Exception:
                    pass

            response.raise_for_status()

            answer = ""
            thinking_content = ""
            prompt_tokens = 0
            completion_tokens = 0
            first_token_time = None
            start_time = time.time()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    # Capture thinking content (for models like DeepSeek-R1)
                    if 'message' in chunk:
                        msg = chunk['message']

                        # Check for thinking content in various possible locations
                        if 'thinking' in msg:
                            thinking_content += msg['thinking']
                            if self.stream_callback:
                                self.stream_callback({
                                    'type': 'thinking',
                                    'content': msg['thinking'],
                                    'name': self.name
                                })

                        # Regular content
                        if 'content' in msg:
                            content = msg['content']
                            answer += content

                            if first_token_time is None and content:
                                first_token_time = time.time() - start_time

                            if self.stream_callback:
                                self.stream_callback({
                                    'type': 'content',
                                    'content': content,
                                    'name': self.name
                                })

                    if chunk.get('done', False):
                        prompt_tokens = chunk.get('prompt_eval_count', 0)
                        completion_tokens = chunk.get('eval_count', 0)

            total = prompt_tokens + completion_tokens
            eval_duration = chunk.get('eval_duration', 0)
            tokens_per_second = 0
            if eval_duration > 0 and completion_tokens > 0:
                eval_duration_sec = eval_duration / 1e9
                tokens_per_second = completion_tokens / eval_duration_sec

            ttft = first_token_time if first_token_time else 0

            # Store thinking content
            self.last_thinking = thinking_content
            if thinking_content:
                self.thinking_history.append({
                    'round': round_num,
                    'phase': phase,
                    'thinking': thinking_content
                })

            # Store message with thinking metadata
            message_entry = {"role": "assistant", "content": answer}
            if thinking_content:
                message_entry["thinking"] = thinking_content
            self.messages.append(message_entry)

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total

            # Estimate thinking vs speaking token split based on character ratio
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

        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with Ollama: {e}"
            if self.stream_callback:
                self.stream_callback({
                    'type': 'error',
                    'error': error_msg,
                    'name': self.name
                })
            raise Exception(error_msg)

    def reset_conversation(self):
        """Reset conversation history and token counts."""
        self.messages = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_thinking_tokens = 0
        self.total_speaking_tokens = 0
        self.thinking_history = []
        self.last_thinking = ""

    def set_system_prompt(self, prompt: str):
        """Set the system prompt for conversations.

        For Ollama, this clears any existing messages and sets a new system message.
        """
        self.messages = [{"role": "system", "content": prompt}]

    def get_full_context(self) -> dict:
        """Get the full context window contents and metadata for this client."""
        total_chars = sum(len(msg.get('content', '')) for msg in self.messages)
        thinking_chars = sum(len(msg.get('thinking', '')) for msg in self.messages if 'thinking' in msg)

        return {
            'name': self.name,
            'model': self.model,
            'host': self.host,
            'role': self.role,
            'thinking_enabled': self.thinking,
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
                'num_ctx': self.num_ctx
            },
            'parameters': {
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'repeat_penalty': self.repeat_penalty,
                'num_predict': self.num_predict,
                'be_brief': self.be_brief,
                'reasoning_effort': self.reasoning_effort
            }
        }
