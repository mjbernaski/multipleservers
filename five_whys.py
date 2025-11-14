#!/usr/bin/env python3
"""
5 Whys Strategy with Ollama
Implements the 5 Whys technique using an Ollama instance.
"""

import requests
import json
import sys
import re
from typing import Dict, List, Tuple


class FiveWhysOllama:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip('/')
        self.model = model
        self.conversation_history: List[Dict] = []
        self.messages: List[Dict] = []  # Full conversation for Ollama context
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.output_lines: List[str] = []  # Collect output for file writing

    def check_server_available(self) -> bool:
        """Check if the Ollama server is available and the model exists."""
        try:
            # First, check if server is reachable
            url = f"{self.host}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            # Check if the specified model exists
            models = response.json().get('models', [])
            model_names = [m.get('name') for m in models]

            if self.model not in model_names:
                print(f"Error: Model '{self.model}' not found on server {self.host}")
                print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                return False

            print(f"✓ Connected to Ollama server at {self.host}")
            print(f"✓ Model '{self.model}' is available")
            return True

        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Ollama server at {self.host}")
            print(f"Please verify:")
            print(f"  1. The server is running")
            print(f"  2. The IP address/port is correct")
            print(f"  3. Network connectivity to {self.host}")
            return False
        except requests.exceptions.Timeout:
            print(f"Error: Connection to {self.host} timed out")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to communicate with Ollama server: {e}")
            return False

    def ask_question(self, question: str) -> Tuple[str, Dict]:
        """Send a question to Ollama and get response with token counts."""
        url = f"{self.host}/api/chat"

        # Add user message to conversation history
        self.messages.append({"role": "user", "content": question})

        # Debug: Print message count being sent
        print(f"[DEBUG: Sending {len(self.messages)} messages to Ollama]", file=sys.stderr)
        if len(self.messages) > 0:
            first_msg_preview = self.messages[0]['content'][:100].replace('\n', ' ')
            print(f"[DEBUG: First message preview: {first_msg_preview}...]", file=sys.stderr)

        payload = {
            "model": self.model,
            "messages": self.messages,  # Send full conversation context
            "stream": True
        }

        try:
            response = requests.post(url, json=payload, timeout=300, stream=True)
            response.raise_for_status()

            answer = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Process streaming response
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    # Get content from this chunk
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        answer += content
                        # Print as we receive it
                        print(content, end='', flush=True)

                    # Token counts come in the final message
                    if chunk.get('done', False):
                        prompt_tokens = chunk.get('prompt_eval_count', 0)
                        completion_tokens = chunk.get('eval_count', 0)

            print()  # New line after streaming completes

            total = prompt_tokens + completion_tokens

            # Add assistant response to conversation history
            self.messages.append({"role": "assistant", "content": answer})

            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total

            token_info = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total': total
            }

            return answer, token_info

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            sys.exit(1)

    def print_and_save(self, text: str, end: str = '\n', flush: bool = False):
        """Print to console and save to output buffer."""
        print(text, end=end, flush=flush)
        self.output_lines.append(text + end)

    def sanitize_filename(self, question: str, max_length: int = 50) -> str:
        """Create a safe filename from the question."""
        # Remove or replace invalid filename characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '', question)
        # Replace spaces and other whitespace with underscores
        safe_name = re.sub(r'\s+', '_', safe_name)
        # Remove any remaining non-alphanumeric characters except underscore and dash
        safe_name = re.sub(r'[^\w\-]', '', safe_name)
        # Truncate to max length
        safe_name = safe_name[:max_length]
        # Remove trailing underscores or dashes
        safe_name = safe_name.rstrip('_-')
        return safe_name if safe_name else "five_whys_output"

    def save_to_file(self, filename: str):
        """Save the output to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(''.join(self.output_lines))

    def load_context(self, context_file: str) -> str:
        """Load context from a text file."""
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Context file '{context_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading context file: {e}")
            sys.exit(1)

    def run_five_whys(self, initial_question: str, context_file: str = None):
        """Execute the 5 Whys technique."""
        self.print_and_save("=" * 80)
        self.print_and_save("5 WHYS ANALYSIS")
        self.print_and_save("=" * 80)
        self.print_and_save(f"Ollama Host: {self.host}")
        self.print_and_save(f"Model: {self.model}")
        if context_file:
            self.print_and_save(f"Context File: {context_file}")
        self.print_and_save("=" * 80)
        self.print_and_save("")

        # Add system instruction for the 5 Whys process
        system_prompt = """You are participating in a 5 Whys analysis. When asked "Why?", you should:
1. First, address any questions you raised in your previous answer - provide clear answers to them
2. Then, explain why the situation occurred by going one level deeper in the analysis
3. Be thorough but concise in your explanations

This approach ensures we clarify uncertainties before diving deeper into root causes.

IMPORTANT: Provide your responses in plain text only. Do NOT use markdown formatting such as headers (##), bold (**), italics (*), bullet points, or any other markdown syntax. Use simple, clear prose."""

        self.messages.append({
            "role": "system",
            "content": system_prompt
        })
        print(f"[DEBUG: Added system prompt to guide 5 Whys process]", file=sys.stderr)

        # Load and add context if provided
        if context_file:
            context = self.load_context(context_file)
            context_length = len(context)
            context_lines = context.count('\n') + 1

            self.print_and_save("CONTEXT:")
            self.print_and_save(f"[Loaded {context_length} characters, {context_lines} lines from {context_file}]")
            self.print_and_save("-" * 80)
            self.print_and_save(context)
            self.print_and_save("-" * 80)
            self.print_and_save("")

            # Add context as initial user message
            full_context_message = f"Here is some context for our discussion:\n\n{context}\n\nPlease use this context to inform your answers."
            self.messages.append({
                "role": "user",
                "content": full_context_message
            })
            print(f"[DEBUG: Added context message to Ollama ({len(full_context_message)} chars)]", file=sys.stderr)
            # Add acknowledgment from assistant
            self.messages.append({
                "role": "assistant",
                "content": "I understand the context. I'll use this information to inform my answers."
            })

        # Ask initial question
        self.print_and_save(f"INITIAL QUESTION:")
        self.print_and_save(f"Q: {initial_question}")
        self.print_and_save("")
        self.print_and_save(f"A: ", end='', flush=True)

        answer, tokens = self.ask_question(initial_question)

        # Add the streamed answer to output
        self.output_lines.append(answer + '\n')

        self.print_and_save(f"[Tokens - Prompt: {tokens['prompt_tokens']}, Completion: {tokens['completion_tokens']}, Total: {tokens['total']}]")
        self.print_and_save("")
        self.print_and_save("-" * 80)
        self.print_and_save("")

        self.conversation_history.append({
            'round': 0,
            'question': initial_question,
            'answer': answer,
            'tokens': tokens
        })

        # Ask "Why?" 5 times
        for i in range(1, 6):
            self.print_and_save(f"WHY #{i}:")
            self.print_and_save(f"Q: Why?")
            self.print_and_save("")
            self.print_and_save(f"A: ", end='', flush=True)

            # Just ask "Why?" - the full conversation context is maintained
            answer, tokens = self.ask_question("Why?")

            # Add the streamed answer to output
            self.output_lines.append(answer + '\n')

            self.print_and_save(f"[Tokens - Prompt: {tokens['prompt_tokens']}, Completion: {tokens['completion_tokens']}, Total: {tokens['total']}]")
            self.print_and_save("")
            self.print_and_save("-" * 80)
            self.print_and_save("")

            self.conversation_history.append({
                'round': i,
                'question': 'Why?',
                'answer': answer,
                'tokens': tokens
            })

        # Print summary
        self.print_and_save("=" * 80)
        self.print_and_save("SUMMARY")
        self.print_and_save("=" * 80)
        self.print_and_save(f"Total Rounds: 6 (1 initial + 5 whys)")
        self.print_and_save(f"Total Prompt Tokens: {self.total_prompt_tokens}")
        self.print_and_save(f"Total Completion Tokens: {self.total_completion_tokens}")
        self.print_and_save(f"Total Tokens: {self.total_tokens}")
        self.print_and_save("=" * 80)
        self.print_and_save("")

        # Print full transcript
        self.print_and_save("=" * 80)
        self.print_and_save("FULL TRANSCRIPT")
        self.print_and_save("=" * 80)
        self.print_and_save("")

        for entry in self.conversation_history:
            if entry['round'] == 0:
                self.print_and_save(f"INITIAL QUESTION:")
            else:
                self.print_and_save(f"WHY #{entry['round']}:")

            self.print_and_save(f"Q: {entry['question']}")
            self.print_and_save(f"A: {entry['answer']}")
            self.print_and_save(f"[Tokens: {entry['tokens']['total']}]")
            self.print_and_save("")

        # Save to file
        filename = self.sanitize_filename(initial_question) + ".txt"
        self.save_to_file(filename)
        self.print_and_save(f"Output saved to: {filename}")


def main():
    OLLAMA_HOST = "http://192.168.5.46:11434"
    MODEL = "gpt-oss:120b"

    if len(sys.argv) < 2:
        print("Usage: python five_whys.py 'Your initial question here' [context_file.txt]")
        print("Example: python five_whys.py 'Why is the sky blue?'")
        print("Example with context: python five_whys.py 'Why did the deployment fail?' deployment_logs.txt")
        sys.exit(1)

    initial_question = sys.argv[1]
    context_file = sys.argv[2] if len(sys.argv) > 2 else None

    analyzer = FiveWhysOllama(OLLAMA_HOST, MODEL)

    # Check if server is available before proceeding
    if not analyzer.check_server_available():
        sys.exit(1)

    print()  # Blank line before starting analysis
    analyzer.run_five_whys(initial_question, context_file)


if __name__ == "__main__":
    main()
