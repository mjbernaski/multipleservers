#!/usr/bin/env python3
"""
5 Whys Strategy with Ollama - Parallel Execution
Runs 5 Whys analysis on multiple Ollama servers in parallel with web interface.
"""

import requests
import json
import sys
import re
import threading
import os
import tempfile
import uuid
import time
from typing import Dict, List, Tuple, Callable, Optional
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import io


class FiveWhysOllama:
    def __init__(self, host: str, model: str, name: str = None, num_ctx: int = 8192, 
                 temperature: float = None, top_p: float = None, top_k: int = None,
                 repeat_penalty: float = None, num_predict: int = None, thinking: bool = False,
                 be_brief: bool = False):
        self.host = host.rstrip('/')
        self.model = model
        self.name = name or f"{host} ({model})"
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.num_predict = num_predict
        self.thinking = thinking
        self.be_brief = be_brief
        self.conversation_history: List[Dict] = []
        self.messages: List[Dict] = []  # Full conversation for Ollama context
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.output_lines: List[str] = []  # Collect output for file writing
        self.stream_callback: Optional[Callable] = None  # Callback for streaming updates
        self.start_time: Optional[float] = None  # Track analysis start time
        self.end_time: Optional[float] = None  # Track analysis end time

    def set_stream_callback(self, callback: Callable):
        """Set a callback function to receive streaming updates."""
        self.stream_callback = callback

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

    def ask_question(self, question: str, round_num: int = 0) -> Tuple[str, Dict]:
        """Send a question to Ollama and get response with token counts."""
        url = f"{self.host}/api/chat"

        # Prepend "Be Brief." if enabled (always add it, even if already present)
        if self.be_brief:
            question = "Be Brief. " + question

        # Add user message to conversation history
        self.messages.append({"role": "user", "content": question})
        
        # Debug: Log the question being sent (truncate if too long)
        question_preview = question[:200] + "..." if len(question) > 200 else question
        debug_log('info', f"Sending question to Ollama (round {round_num}), length={len(question)} chars", server=self.name, data={'round': round_num, 'question_length': len(question), 'preview': question_preview})

        payload = {
            "model": self.model,
            "messages": self.messages,  # Send full conversation context
            "stream": True
        }
        
        # Add thinking/reasoning mode if enabled
        if self.thinking:
            payload["thinking"] = True
        
        # Add generation parameters if they are set
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
        
        # Debug: Log message count and parameters being sent
        payload_info = {'model': self.model, 'stream': True}
        if self.thinking:
            payload_info['thinking'] = True
        if self.num_ctx is not None:
            payload_info['num_ctx'] = self.num_ctx
        if self.temperature is not None:
            payload_info['temperature'] = self.temperature
        if self.top_p is not None:
            payload_info['top_p'] = self.top_p
        if self.top_k is not None:
            payload_info['top_k'] = self.top_k
        if self.repeat_penalty is not None:
            payload_info['repeat_penalty'] = self.repeat_penalty
        if self.num_predict is not None:
            payload_info['num_predict'] = self.num_predict
        debug_log('debug', f"Sending {len(self.messages)} messages to Ollama", server=self.name, data={'message_count': len(self.messages), 'payload': payload_info})
        if len(self.messages) > 0:
            # Show first message (system prompt) preview
            first_msg = self.messages[0]
            first_preview = first_msg.get('content', '')[:200] + "..." if len(first_msg.get('content', '')) > 200 else first_msg.get('content', '')
            debug_log('debug', f"First message (role={first_msg.get('role')}): {first_preview}", server=self.name)
            # Log all messages for debugging
            for i, msg in enumerate(self.messages):
                role = msg.get('role', 'unknown')
                content_preview = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                debug_log('debug', f"Message {i}: role={role}, length={len(msg.get('content', ''))} chars", server=self.name, data={'preview': content_preview})

        try:
            debug_log('debug', f"POST request to {url}", server=self.name, data={'model': self.model, 'round': round_num})
            response = requests.post(url, json=payload, timeout=300, stream=True)
            response.raise_for_status()
            debug_log('info', f"Response received from Ollama (round {round_num})", server=self.name)

            answer = ""
            prompt_tokens = 0
            completion_tokens = 0
            chunk_count = 0

            # Process streaming response
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    chunk_count += 1

                    # Get content from this chunk
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        answer += content
                        # Call stream callback if set
                        if self.stream_callback:
                            self.stream_callback({
                                'type': 'content',
                                'round': round_num,
                                'content': content,
                                'name': self.name
                            })

                    # Token counts come in the final message
                    if chunk.get('done', False):
                        prompt_tokens = chunk.get('prompt_eval_count', 0)
                        completion_tokens = chunk.get('eval_count', 0)

            total = prompt_tokens + completion_tokens

            debug_log('info', f"Response complete (round {round_num}): {len(answer)} chars, {chunk_count} chunks, tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total} total", server=self.name, data={'answer_length': len(answer), 'chunks': chunk_count, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total})
            debug_log('debug', f"Answer preview (first 200 chars): {answer[:200]}...", server=self.name)

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

            # Notify completion
            if self.stream_callback:
                self.stream_callback({
                    'type': 'round_complete',
                    'round': round_num,
                    'answer': answer,
                    'tokens': token_info,
                    'name': self.name
                })

            return answer, token_info

        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with Ollama: {e}"
            debug_log('error', error_msg, server=self.name, data={'error': str(e), 'round': round_num})
            if self.stream_callback:
                self.stream_callback({
                    'type': 'error',
                    'error': error_msg,
                    'name': self.name
                })
            raise Exception(error_msg)

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

    def load_context(self, context_file: str) -> str:
        """Load context from a text file."""
        try:
            debug_log('debug', f"Loading context file: {context_file}, exists: {os.path.exists(context_file)}", server=self.name)
            if not os.path.exists(context_file):
                raise FileNotFoundError(f"Context file '{context_file}' does not exist")
            with open(context_file, 'r', encoding='utf-8') as f:
                content = f.read()
                debug_log('info', f"Loaded {len(content)} characters from context file", server=self.name, data={'file_size': len(content)})
                return content
        except FileNotFoundError as e:
            error_msg = f"Error: Context file '{context_file}' not found: {e}"
            debug_log('error', error_msg, server=self.name)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error reading context file '{context_file}': {e}"
            debug_log('error', error_msg, server=self.name)
            raise Exception(error_msg)

    def reset_conversation(self):
        """Reset conversation history and token counts."""
        self.conversation_history = []
        self.messages = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def run_five_whys(self, initial_question: str, context_file: str = None, restart: bool = False, continue_from_round: int = None, be_brief: bool = None):
        """Execute the 5 Whys technique.
        
        Args:
            initial_question: The question to ask
            context_file: Optional context file path
            restart: If True, reset conversation history before starting
            continue_from_round: If provided, continue from this round number (0-based)
        """
        # Reset if restart requested
        if restart:
            self.reset_conversation()

        # Update be_brief setting if provided
        if be_brief is not None:
            self.be_brief = be_brief

        # Track start time
        self.start_time = time.time()

        # Notify start
        if self.stream_callback:
            self.stream_callback({
                'type': 'start',
                'name': self.name,
                'host': self.host,
                'model': self.model,
                'question': initial_question
            })

        # Load context first if provided, so we can include it in system prompt
        context_content = None
        debug_log('debug', f"run_five_whys called with context_file='{context_file}'", server=self.name)
        if context_file:
            debug_log('debug', f"About to load context from '{context_file}', exists={os.path.exists(context_file)}", server=self.name)
            try:
                context_content = self.load_context(context_file)
                debug_log('info', f"Successfully loaded context, length={len(context_content) if context_content else 0} characters", server=self.name, data={'file_size': len(context_content) if context_content else 0})
                
                # Notify that context file is being loaded
                if self.stream_callback:
                    self.stream_callback({
                        'type': 'context_file',
                        'name': self.name,
                        'content': context_content,
                        'filename': os.path.basename(context_file) if context_file else 'context.txt',
                        'success': True
                    })
            except Exception as e:
                # Notify error loading context file
                if self.stream_callback:
                    self.stream_callback({
                        'type': 'context_file',
                        'name': self.name,
                        'filename': os.path.basename(context_file) if context_file else 'context.txt',
                        'success': False,
                        'error': str(e)
                    })
                raise

        # Initialize system prompt only if starting fresh
        # Also update system prompt if context is provided (even if analyzer is reused)
        needs_system_prompt = restart or len(self.messages) == 0
        if needs_system_prompt or (context_content and len(self.messages) > 0):
            # Build system prompt with context if available
            system_prompt = """You are participating in a 5 Whys analysis. When asked "Why?", you should:
1. First, address any questions you raised in your previous answer - provide clear answers to them
2. Then, explain why the situation occurred by going one level deeper in the analysis
3. Be thorough but concise in your explanations

This approach ensures we clarify uncertainties before diving deeper into root causes.

IMPORTANT: Provide your responses in plain text only. Do NOT use markdown formatting such as headers (##), bold (**), italics (*), bullet points, or any other markdown syntax. Use simple, clear prose."""

            # Include context in system prompt if available
            if context_content:
                system_prompt = f"""{system_prompt}

CRITICAL: You have been provided with context content that will be included in the user's messages. You MUST:
1. Read and analyze ALL context content provided to you
2. Use the context information to inform your answers
3. When referencing specific information from the context, provide citations in the format: [FILE: filename.txt, ID: identifier] or [Context: specific quote or reference]
4. If the user asks about "this writing" or "this content", they are referring to the context that was provided
5. NEVER say "no writing was provided" or "no content was provided" - the context IS the writing/content

The context will be explicitly provided in the user's message. Pay close attention to it."""
            
            if needs_system_prompt:
                # Add new system prompt
                self.messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            elif context_content and len(self.messages) > 0:
                # Update existing system prompt if context is provided but analyzer is reused
                # Find and update the system message if it exists
                for i, msg in enumerate(self.messages):
                    if msg.get("role") == "system":
                        self.messages[i] = {"role": "system", "content": system_prompt}
                        break
                else:
                    # No system message found, add one at the beginning
                    self.messages.insert(0, {"role": "system", "content": system_prompt})

        # Determine starting round
        # If context_file is provided, always start from round 0 to include context with the new question
        # This ensures context is always included when a new question is asked with a file
        start_round = 0
        if continue_from_round is not None:
            start_round = continue_from_round + 1
        elif not restart and len(self.conversation_history) > 0 and not context_file:
            # Continue from where we left off ONLY if no context file is provided
            # If we've completed all 6 rounds, start a new round 0 with the new question
            if len(self.conversation_history) >= 6:
                start_round = 0
            else:
                start_round = len(self.conversation_history)
        # If context_file is provided, we always start from round 0 to include it with the question

        # Ask initial question if starting from round 0
        if start_round == 0:
            # If context exists, combine it with the question so "this" references work
            if context_content:
                debug_log('info', f"Combining context ({len(context_content)} chars) with question for initial round", server=self.name, data={'context_length': len(context_content), 'question_length': len(initial_question)})
                debug_log('debug', f"Context preview (first 200 chars): {context_content[:200]}...", server=self.name)
                # Combine context and question into one message so references are clear
                # Make it very explicit that this IS the writing/content being referenced
                combined_question = f"""IMPORTANT: The following is the writing/content you need to analyze:

{context_content}

Now, please answer this question about the above content:

{initial_question}

Remember: The content above IS the writing/content being referenced. Use it to answer the question and provide citations when referencing specific parts."""
                debug_log('debug', f"Combined question length: {len(combined_question)} chars", server=self.name, data={'combined_length': len(combined_question)})
            else:
                debug_log('warning', f"No context content, using question as-is. context_file was '{context_file}', context_content is None", server=self.name)
                combined_question = initial_question
            
            if self.stream_callback:
                self.stream_callback({
                    'type': 'question',
                    'round': 0,
                    'question': initial_question,
                    'name': self.name
                })

            answer, tokens = self.ask_question(combined_question, round_num=0)
            self.conversation_history.append({
                'round': 0,
                'question': initial_question,
                'answer': answer,
                'tokens': tokens
            })
            start_round = 1

        # Ask "Why?" for remaining rounds (up to 5 total whys)
        total_rounds = 6  # 1 initial + 5 whys
        for i in range(start_round, total_rounds):
            if self.stream_callback:
                self.stream_callback({
                    'type': 'question',
                    'round': i,
                    'question': 'Why?',
                    'name': self.name
                })

            answer, tokens = self.ask_question("Why?", round_num=i)
            self.conversation_history.append({
                'round': i,
                'question': 'Why?',
                'answer': answer,
                'tokens': tokens
            })

        # Track end time and calculate runtime
        self.end_time = time.time()
        runtime_seconds = self.end_time - self.start_time if self.start_time else 0

        # Notify completion
        if self.stream_callback:
            self.stream_callback({
                'type': 'complete',
                'name': self.name,
                'total_prompt_tokens': self.total_prompt_tokens,
                'total_completion_tokens': self.total_completion_tokens,
                'total_tokens': self.total_tokens,
                'conversation_history': self.conversation_history,
                'runtime_seconds': runtime_seconds
            })


# Global state for web interface
app = Flask(__name__)
app.config['SECRET_KEY'] = 'five-whys-secret-key'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")

# Store uploaded files temporarily (session_id -> filepath)
uploaded_files = {}

# Debug logging function - emits to WebSocket and also prints
def debug_log(level, message, server=None, data=None):
    """Emit debug log event via WebSocket and also print to console."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    log_entry = {
        'timestamp': timestamp,
        'level': level,  # 'info', 'warning', 'error', 'debug'
        'message': message,
        'server': server
    }
    if data:
        log_entry['data'] = data
    
    # Emit to WebSocket for debug page
    socketio.emit('debug_log', log_entry)
    
    # Also print to console
    server_prefix = f"[{server}] " if server else ""
    print(f"[{timestamp}] [{level.upper()}] {server_prefix}{message}", flush=True)

# Track file usage count (session_id -> count) to prevent premature deletion
file_usage_count = {}

# Store analyzer instances to maintain context between requests (server_name -> analyzer)
analyzer_instances = {}

# Store analysis metadata for PDF generation
analysis_metadata = {}  # analysis_id -> {question, context_filename, timestamp, servers}

# Store runtime for each server
server_runtimes = {}  # analysis_id -> {server_name: runtime_seconds}


def run_analysis_thread(analyzer: FiveWhysOllama, question: str, context_file: str, name: str, session_id: str = None, restart: bool = False, be_brief: bool = False):
    """Run analysis in a separate thread."""
    try:
        debug_log('info', f"run_analysis_thread for {name}: context_file={context_file}, exists={os.path.exists(context_file) if context_file else False}, session_id={session_id}, be_brief={be_brief}", server=name, data={'context_file': context_file, 'session_id': session_id, 'be_brief': be_brief})
        if context_file:
            file_size = os.path.getsize(context_file) if os.path.exists(context_file) else 0
            debug_log('info', f"Context file path for {name}: '{context_file}', file size: {file_size} bytes", server=name, data={'path': context_file, 'file_size': file_size})
        analyzer.run_five_whys(question, context_file, restart=restart, be_brief=be_brief)
    except Exception as e:
        socketio.emit('error', {
            'server': name,
            'error': str(e)
        })
    finally:
        # Decrement usage count and clean up if this was the last thread using the file
        if session_id and session_id in file_usage_count:
            file_usage_count[session_id] -= 1
            if file_usage_count[session_id] <= 0:
                # Last thread finished, safe to delete
                if session_id in uploaded_files:
                    temp_path = uploaded_files.pop(session_id)
                    if os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass  # Ignore cleanup errors
                # Clean up usage count
                if session_id in file_usage_count:
                    del file_usage_count[session_id]


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('five_whys_parallel.html')


@app.route('/debug')
def debug():
    """Serve the debug log page."""
    return render_template('debug.html')


def format_runtime(seconds):
    """Format runtime in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} {secs:.1f} second{'s' if secs != 1 else ''}"


def generate_text_file(server_name, analysis_id):
    """Generate a text file with the complete dialog for a server."""
    if analysis_id not in analysis_metadata:
        return None
    
    if server_name not in analyzer_instances:
        return None
    
    metadata = analysis_metadata[analysis_id]
    analyzer = analyzer_instances[server_name]
    
    # Create safe filename
    safe_question = re.sub(r'[<>:"/\\|?*]', '', metadata['question'])[:50]
    safe_question = safe_question.replace(' ', '_')
    safe_server = re.sub(r'[<>:"/\\|?*]', '', server_name)[:30]
    safe_server = safe_server.replace(' ', '_').replace('(', '').replace(')', '')
    
    timestamp_str = metadata['timestamp'][:19].replace('T', '_').replace(':', '-')
    filename = f"{safe_server}_{safe_question}_{timestamp_str}.txt"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Get runtime
    runtime_seconds = server_runtimes.get(analysis_id, {}).get(server_name, 0)
    runtime_str = format_runtime(runtime_seconds)
    
    # Write the file
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("5 Whys Analysis - Complete Dialog\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Server: {server_name}\n")
        f.write(f"Host: {analyzer.host}\n")
        f.write(f"Model: {analyzer.model}\n")
        f.write(f"Initial Question: {metadata['question']}\n")
        if metadata['context_filename']:
            f.write(f"Context File: {metadata['context_filename']}\n")
        f.write(f"Date: {metadata['timestamp'][:19].replace('T', ' ')}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write each round
        for i, round_data in enumerate(analyzer.conversation_history):
            round_label = "INITIAL QUESTION" if i == 0 else f"WHY #{i}"
            f.write(f"\n{round_label}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question: {round_data.get('question', '')}\n\n")
            f.write(f"Answer:\n{round_data.get('answer', '')}\n\n")
            
            tokens = round_data.get('tokens', {})
            if tokens:
                f.write(f"Tokens - Prompt: {tokens.get('prompt_tokens', 0)}, ")
                f.write(f"Completion: {tokens.get('completion_tokens', 0)}, ")
                f.write(f"Total: {tokens.get('total', 0)}\n")
            f.write("\n")
        
        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Rounds: {len(analyzer.conversation_history)} (1 initial + 5 whys)\n")
        f.write(f"Total Prompt Tokens: {analyzer.total_prompt_tokens}\n")
        f.write(f"Total Completion Tokens: {analyzer.total_completion_tokens}\n")
        f.write(f"Total Tokens: {analyzer.total_tokens}\n")
        f.write(f"Runtime: {runtime_str}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    debug_log('info', f"Generated text file: {filename}", server=server_name, data={'filepath': filepath, 'runtime': runtime_str})
    
    return filepath


@app.route('/generate_pdf/<analysis_id>')
def generate_pdf(analysis_id):
    """Generate a side-by-side PDF of the 5 Whys analysis."""
    if analysis_id not in analysis_metadata:
        return jsonify({'error': 'Analysis not found'}), 404
    
    metadata = analysis_metadata[analysis_id]
    servers = metadata['servers']
    
    # Get analyzer instances for the servers
    analyzers_data = []
    for server_name in servers:
        if server_name in analyzer_instances:
            analyzer = analyzer_instances[server_name]
            runtime_seconds = server_runtimes.get(analysis_id, {}).get(server_name, 0)
            analyzers_data.append({
                'name': server_name,
                'host': analyzer.host,
                'model': analyzer.model,
                'conversation_history': analyzer.conversation_history,
                'total_prompt_tokens': analyzer.total_prompt_tokens,
                'total_completion_tokens': analyzer.total_completion_tokens,
                'total_tokens': analyzer.total_tokens,
                'runtime_seconds': runtime_seconds
            })
    
    if len(analyzers_data) == 0:
        return jsonify({'error': 'No analysis data found'}), 404
    
    # Create PDF in memory with better margins
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                          rightMargin=0.6*inch, leftMargin=0.6*inch,
                          topMargin=0.6*inch, bottomMargin=0.6*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles with Excel color palette
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f4e78'),
        spaceAfter=16,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        leading=24
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1f4e78'),
        spaceAfter=10,
        spaceBefore=16,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        leading=18
    )
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#1f4e78'),
        spaceAfter=4,
        fontName='Helvetica-Bold',
        leading=12
    )
    
    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontSize=9.5,
        textColor=colors.HexColor('#000000'),
        spaceAfter=10,
        alignment=TA_JUSTIFY,
        leading=13.5,
        leftIndent=0,
        rightIndent=0
    )
    
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#808080'),
        spaceAfter=3,
        leading=11
    )
    
    token_style = ParagraphStyle(
        'TokenStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#808080'),
        spaceAfter=0,
        leading=10,
        fontName='Helvetica-Oblique'
    )
    
    # Title
    elements.append(Paragraph("5 Whys Analysis - Side by Side Comparison", title_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Analysis metadata in a subtle box - escape HTML
    question_escaped = metadata['question'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    metadata_text = f"<b>Initial Question:</b> {question_escaped}<br/>"
    if metadata['context_filename']:
        filename_escaped = metadata['context_filename'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        metadata_text += f"<b>Context File:</b> {filename_escaped}<br/>"
    metadata_text += f"<b>Date:</b> {metadata['timestamp'][:19].replace('T', ' ')}"
    
    # Create a subtle background for metadata with Excel colors
    metadata_table = Table([[Paragraph(metadata_text, info_style)]], colWidths=[doc.width])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f2f2f2')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d0d0')),
    ]))
    elements.append(metadata_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Get maximum number of rounds
    max_rounds = max(len(a['conversation_history']) for a in analyzers_data) if analyzers_data else 0
    
    # Create side-by-side content for each round
    for round_num in range(max_rounds):
        round_label = "Initial Question" if round_num == 0 else f"Why #{round_num}"
        elements.append(Paragraph(f"<b>{round_label}</b>", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Create table for side-by-side comparison
        table_data = []
        
        # Header row
        header_row = []
        for analyzer_data in analyzers_data:
            header_para_style = ParagraphStyle(
                'HeaderStyle',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#ffffff'),
                alignment=TA_LEFT,
                fontName='Helvetica-Bold',
                leading=12
            )
            header_row.append(Paragraph(
                f"<b>{analyzer_data['name']}</b><br/><font size=7>{analyzer_data['model']}</font>",
                header_para_style
            ))
        table_data.append(header_row)
        
        # Content row
        content_row = []
        for analyzer_data in analyzers_data:
            round_content = ""
            if round_num < len(analyzer_data['conversation_history']):
                round_data = analyzer_data['conversation_history'][round_num]
                question = round_data.get('question', '')
                answer = round_data.get('answer', '')
                tokens = round_data.get('tokens', {})
                
                # Format question and answer with better styling - escape HTML in content
                # Escape special characters for XML/HTML
                question_escaped = question.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                answer_escaped = answer.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                round_content = f"<b>Q:</b> {question_escaped}<br/><br/>{answer_escaped}"
                if tokens:
                    token_text = f"Tokens: {tokens.get('prompt_tokens', 0)} prompt + {tokens.get('completion_tokens', 0)} completion = {tokens.get('total', 0)} total"
                    round_content += f"<br/><br/><font color='#808080' size=7><i>{token_text}</i></font>"
            else:
                round_content = "<i>No data for this round</i>"
            
            # Use a container paragraph style for better formatting
            content_para_style = ParagraphStyle(
                'ContentStyle',
                parent=answer_style,
                fontSize=9.5,
                textColor=colors.HexColor('#000000'),
                leading=13.5
            )
            content_row.append(Paragraph(round_content, content_para_style))
        
        table_data.append(content_row)
        
        # Create table with earth-tone styling
        col_widths = [doc.width / len(analyzers_data)] * len(analyzers_data)
        table = Table(table_data, colWidths=col_widths, hAlign='LEFT')
        table.setStyle(TableStyle([
            # Header row styling with Excel blue
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 14),
            ('TOPPADDING', (0, 0), (-1, 0), 14),
            # Content row styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d0d0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 12),
            # Inner borders for better separation
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#1f4e78')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
    
    # Summary section
    elements.append(PageBreak())
    elements.append(Paragraph("Summary", heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    summary_data = [['Server', 'Model', 'Total Prompt Tokens', 'Total Completion Tokens', 'Total Tokens', 'Runtime']]
    for analyzer_data in analyzers_data:
        runtime_str = format_runtime(analyzer_data.get('runtime_seconds', 0))
        summary_data.append([
            analyzer_data['name'],
            analyzer_data['model'],
            str(analyzer_data['total_prompt_tokens']),
            str(analyzer_data['total_completion_tokens']),
            str(analyzer_data['total_tokens']),
            runtime_str
        ])
    
    summary_table = Table(summary_data, colWidths=[1.8*inch, 1.3*inch, 1.1*inch, 1.3*inch, 0.9*inch, 1.2*inch])
    summary_table.setStyle(TableStyle([
        # Header styling with Excel blue
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#ffffff')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        # Row styling - alternating for better readability
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f2f2f2')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d0d0')),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#1f4e78')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#000000')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer
    buffer.seek(0)
    
    # Create filename
    safe_question = re.sub(r'[<>:"/\\|?*]', '', metadata['question'])[:50]
    filename = f"5_whys_analysis_{safe_question}_{analysis_id[:8]}.pdf"
    filename = filename.replace(' ', '_')
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=False,  # Open in browser instead of forcing download
        download_name=filename
    )


@app.route('/check_servers', methods=['POST'])
def check_servers():
    """Check the status of configured servers."""
    data = request.get_json()
    servers = data.get('servers', [])
    
    server_statuses = []
    for server_config in servers:
        host = server_config.get('host')
        model = server_config.get('model')
        name = server_config.get('name', f"{host} ({model})")
        
        analyzer = FiveWhysOllama(host, model, name)
        is_available = analyzer.check_server_available()
        
        server_statuses.append({
            'name': name,
            'host': host,
            'model': model,
            'available': is_available
        })
    
    return jsonify({'servers': server_statuses})


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return the file path."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Create a temporary file
    import uuid
    temp_path = None
    try:
        # Create a temporary file with a unique name
        suffix = '_' + secure_filename(file.filename) if file.filename else ''
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            # Write file content directly
            file.seek(0)  # Reset file pointer
            tmp_file.write(file.read())
        
        # Verify file was written
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise Exception('File was not saved correctly')
        
        # Store the file path with a session ID
        session_id = str(uuid.uuid4())
        uploaded_files[session_id] = temp_path
        debug_log('info', f"File uploaded: {file.filename} -> {temp_path} (session_id: {session_id})", data={'filename': file.filename, 'path': temp_path, 'session_id': session_id, 'size': os.path.getsize(temp_path)})
        return jsonify({'session_id': session_id, 'filename': file.filename})
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500


@socketio.on('start_analysis')
def handle_start_analysis(data):
    """Start parallel 5 Whys analysis on multiple servers."""
    question = data.get('question')
    context_file = data.get('context_file')  # Can be a path or session_id
    session_id = data.get('session_id')  # For uploaded files
    servers = data.get('servers', [])
    restart = data.get('restart', False)  # Whether to restart the dialog
    be_brief = data.get('be_brief', False)  # Whether to include "Be Brief." in questions
    
    debug_log('info', f"handle_start_analysis received: question='{question}', context_file='{context_file}', session_id='{session_id}', restart={restart}, be_brief={be_brief}")
    debug_log('debug', f"Current uploaded_files keys: {list(uploaded_files.keys())}")
    
    # If session_id is provided, get the uploaded file path
    if session_id and session_id in uploaded_files:
        context_file = uploaded_files[session_id]
        debug_log('info', f"Using uploaded file: session_id={session_id}, path={context_file}, exists={os.path.exists(context_file) if context_file else False}", data={'session_id': session_id, 'path': context_file})
        if context_file and os.path.exists(context_file):
            file_size = os.path.getsize(context_file)
            debug_log('info', f"Uploaded file size: {file_size} bytes", data={'file_size': file_size})
        else:
            debug_log('error', f"Uploaded file path '{context_file}' does not exist!", data={'path': context_file})
            context_file = None  # Clear invalid path
    elif session_id:
        debug_log('warning', f"session_id {session_id} not found in uploaded_files!", data={'session_id': session_id, 'available_ids': list(uploaded_files.keys())})
        # Don't use context_file from text input if session_id was provided but not found
        context_file = None
    else:
        debug_log('debug', f"No session_id provided, using context_file='{context_file}' if provided")
        # If context_file is provided via text input, verify it exists
        if context_file and not os.path.exists(context_file):
            debug_log('warning', f"context_file '{context_file}' from text input does not exist!", data={'path': context_file})
            context_file = None

    if not question:
        emit('error', {'error': 'No question provided'})
        return

    if not servers:
        emit('error', {'error': 'No servers configured'})
        return

    # Store analysis metadata for PDF generation (create analysis_id early)
    import datetime
    analysis_id = str(uuid.uuid4())
    analysis_metadata[analysis_id] = {
        'question': question,
        'context_filename': os.path.basename(context_file) if context_file else None,
        'context_file': context_file,
        'timestamp': datetime.datetime.now().isoformat(),
        'servers': []  # Will be populated below
    }
    server_runtimes[analysis_id] = {}

    # Get thinking mode and generation parameters from request (optional)
    thinking_params = data.get('thinking_params', {})
    thinking_mode = thinking_params.get('thinking', False)
    num_ctx = thinking_params.get('num_ctx')
    temperature = thinking_params.get('temperature')
    top_p = thinking_params.get('top_p')
    top_k = thinking_params.get('top_k')
    repeat_penalty = thinking_params.get('repeat_penalty')
    num_predict = thinking_params.get('num_predict')

    # Create and validate analyzers for each server
    analyzers = []
    for server_config in servers:
        host = server_config.get('host')
        model = server_config.get('model')
        name = server_config.get('name', f"{host} ({model})")

        # Reuse existing analyzer if available and not restarting
        is_new_analyzer = name not in analyzer_instances
        if not restart and not is_new_analyzer:
            analyzer = analyzer_instances[name]
            # Update thinking mode and generation parameters if provided
            analyzer.thinking = thinking_mode
            analyzer.be_brief = be_brief
            if temperature is not None:
                analyzer.temperature = temperature
            if top_p is not None:
                analyzer.top_p = top_p
            if top_k is not None:
                analyzer.top_k = top_k
            if repeat_penalty is not None:
                analyzer.repeat_penalty = repeat_penalty
            if num_predict is not None:
                analyzer.num_predict = num_predict
            if num_ctx is not None:
                analyzer.num_ctx = num_ctx
        else:
            # Create new analyzer with thinking mode and generation parameters
            analyzer = FiveWhysOllama(
                host, model, name,
                num_ctx=num_ctx if num_ctx is not None else 8192,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                num_predict=num_predict,
                thinking=thinking_mode,
                be_brief=be_brief
            )
            analyzer_instances[name] = analyzer

        # Set up callback to emit WebSocket events
        def make_callback(server_name, analysis_id_ref):
            def callback(event_data):
                # Track runtime when analysis completes
                if event_data.get('type') == 'complete' and 'runtime_seconds' in event_data:
                    if analysis_id_ref in server_runtimes:
                        server_runtimes[analysis_id_ref][server_name] = event_data['runtime_seconds']
                        # Auto-generate text file when server completes
                        generate_text_file(server_name, analysis_id_ref)
                
                socketio.emit('analysis_update', {
                    'server': server_name,
                    **event_data
                })
            return callback

        analyzer.set_stream_callback(make_callback(name, analysis_id))

        # Check server availability (always check for new analyzers, or when restarting)
        if is_new_analyzer or restart:
            if not analyzer.check_server_available():
                emit('error', {
                    'server': name,
                    'error': f'Server {host} or model {model} not available'
                })
                continue

        analyzers.append((analyzer, name))

    # Update servers list in metadata now that we have the final analyzers
    analysis_metadata[analysis_id]['servers'] = [name for _, name in analyzers]

    # Track file usage count if we have a session_id (uploaded file)
    if session_id and session_id in uploaded_files:
        # Track how many threads will use this file (based on actual analyzers, not all servers)
        file_usage_count[session_id] = len(analyzers)
    
    # Emit analysis_started BEFORE starting threads so client can initialize
    # Include context file info if provided
    analysis_started_data = {
        'servers': [name for _, name in analyzers],
        'has_context_file': bool(context_file),
        'context_filename': os.path.basename(context_file) if context_file else None,
        'analysis_id': analysis_id
    }
    emit('analysis_started', analysis_started_data)

    # Now start analysis threads
    threads = []
    for analyzer, name in analyzers:
        thread = threading.Thread(
            target=run_analysis_thread,
            args=(analyzer, question, context_file, name, session_id if session_id else None, restart, be_brief),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Log context file info for debugging
    if context_file:
        file_exists = os.path.exists(context_file) if context_file else False
        file_size = os.path.getsize(context_file) if file_exists else 0
        debug_log('info', f"Context file '{context_file}' will be used by {len(analyzers)} analyzer(s)", data={'path': context_file, 'analyzers': len(analyzers), 'exists': file_exists, 'size': file_size})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='5 Whys Analysis - Parallel Web Interface')
    parser.add_argument('--port', type=int, default=5005, help='Port to run web server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()

    print(f"Starting web server on http://{args.host}:{args.port}")
    print("Open this URL in your browser to view the parallel 5 Whys analysis")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
