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

    def __init__(self, host: str, model: str, name: str = None, num_ctx: int = 8192,
                 temperature: float = None, top_p: float = None, top_k: int = None,
                 repeat_penalty: float = None, num_predict: int = None, thinking: bool = False,
                 be_brief: bool = False, keep_alive: str = "10m"):
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
            thinking: Enable thinking mode
            be_brief: Prepend "Be Brief. " to all prompts
            keep_alive: Model keep-alive duration
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
        self.be_brief = be_brief
        self.keep_alive = keep_alive

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

    def ask(self, question: str, round_num: int = 0) -> Tuple[str, Dict]:
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
            prompt_tokens = 0
            completion_tokens = 0
            first_token_time = None
            start_time = time.time()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
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

            self.messages.append({"role": "assistant", "content": answer})

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total

            token_info = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total': total,
                'tokens_per_second': tokens_per_second,
                'time_to_first_token': ttft
            }

            if self.stream_callback:
                self.stream_callback({
                    'type': 'response_complete',
                    'answer': answer,
                    'tokens': token_info,
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
