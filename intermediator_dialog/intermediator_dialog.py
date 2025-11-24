#!/usr/bin/env python3
"""
Intermediator Dialog System
One AI intermediates a dialog between two other AIs.
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
from dotenv import load_dotenv
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client for TTS
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class OllamaClient:
    """Client for communicating with Ollama servers."""
    
    def __init__(self, host: str, model: str, name: str = None, num_ctx: int = 8192,
                 temperature: float = None, top_p: float = None, top_k: int = None,
                 repeat_penalty: float = None, num_predict: int = None, thinking: bool = False,
                 be_brief: bool = False, keep_alive: str = "10m"):
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
        self.keep_alive = keep_alive
        self.messages: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.stream_callback: Optional[Callable] = None

    def set_stream_callback(self, callback: Callable):
        """Set a callback function to receive streaming updates."""
        self.stream_callback = callback

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

                    # Handle thinking tokens separately (don't include in answer/TTS)
                    if 'message' in chunk and 'thinking' in chunk['message']:
                        thinking_content = chunk['message']['thinking']
                        # Optionally log thinking for debugging
                        # print(f"[THINKING] {thinking_content}")

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


class IntermediatorDialog:
    """Manages a dialog between two AIs with an intermediator."""

    def __init__(self, intermediator: OllamaClient, participant1: OllamaClient, participant2: OllamaClient,
                 intermediator_pre_prompt: str = None,
                 intermediator_topic_prompt: str = None,
                 participant_pre_prompt: str = None,
                 participant1_mid_prompt: str = None,
                 participant2_mid_prompt: str = None,
                 participant_post_prompt: str = None,
                 dialog_id: str = None,
                 enable_tts: bool = True):
        self.intermediator = intermediator
        self.participant1 = participant1
        self.participant2 = participant2
        self.intermediator_pre_prompt = intermediator_pre_prompt
        self.intermediator_topic_prompt = intermediator_topic_prompt
        self.participant_pre_prompt = participant_pre_prompt
        self.participant1_mid_prompt = participant1_mid_prompt
        self.participant2_mid_prompt = participant2_mid_prompt
        self.participant_post_prompt = participant_post_prompt
        self.conversation_history: List[Dict] = []
        self.current_turn = 0
        self.turn_counter = 0  # Sequential counter for unique turn numbers
        self.stream_callback: Optional[Callable] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.dialog_id = dialog_id or str(uuid.uuid4())
        self.enable_tts = enable_tts
        self.audio_sequence = 0  # Global sequence counter for TTS audio files
        self.tts_threads: List[threading.Thread] = []  # Track TTS threads to ensure they complete

    def set_stream_callback(self, callback: Callable):
        """Set a callback function to receive streaming updates."""
        self.stream_callback = callback
        self.intermediator.set_stream_callback(callback)
        self.participant1.set_stream_callback(callback)
        self.participant2.set_stream_callback(callback)

    def _emit(self, event_type: str, data: Dict):
        """Emit event via callback."""
        if self.stream_callback:
            self.stream_callback({
                'type': event_type,
                **data
            })

    def run_dialog(self, max_turns: int = 3, context_file: str = None):
        """Run a dialog between the two participants, mediated by the intermediator."""
        self.start_time = time.time()
        
        # Load context if provided
        context_content = None
        if context_file and os.path.exists(context_file):
            with open(context_file, 'r', encoding='utf-8') as f:
                context_content = f.read()

        # Initialize system prompts
        # Build intermediator prompt from pre-prompt + topic prompt
        no_markdown_instruction = "\n\nIMPORTANT: Use plain text only. Do NOT use markdown formatting (no **bold**, no *italic*, no # headers, no lists with markdown syntax, etc.). Write in natural, plain text."
        
        if self.intermediator_pre_prompt and self.intermediator_topic_prompt:
            intermediator_system_prompt = f"{self.intermediator_pre_prompt}\n\n{self.intermediator_topic_prompt}{no_markdown_instruction}"
        elif self.intermediator_topic_prompt:
            # If only topic prompt provided, use default pre-prompt
            default_pre = """You are a thoughtful moderator facilitating a dialog between two AI participants. Your role is to:
1. Guide the conversation to explore the topic deeply
2. Ask clarifying questions when needed
3. Summarize key points when appropriate
4. Ensure both participants have opportunities to contribute
5. Keep the conversation focused and productive

Be concise but insightful in your moderation."""
            intermediator_system_prompt = f"{default_pre}\n\n{self.intermediator_topic_prompt}{no_markdown_instruction}"
        elif self.intermediator_pre_prompt:
            # If only pre-prompt provided, use it alone
            intermediator_system_prompt = f"{self.intermediator_pre_prompt}{no_markdown_instruction}"
        else:
            # Default prompt if nothing provided
            intermediator_system_prompt = """You are a thoughtful moderator facilitating a dialog between two AI participants. Your role is to:
1. Guide the conversation to explore the topic deeply
2. Ask clarifying questions when needed
3. Summarize key points when appropriate
4. Ensure both participants have opportunities to contribute
5. Keep the conversation focused and productive

Be concise but insightful in your moderation.""" + no_markdown_instruction
        
        # Build participant prompts from pre-prompt + mid-prompt + post-prompt
        def build_participant_prompt(mid_prompt: str = None) -> str:
            parts = []
            if self.participant_pre_prompt:
                parts.append(self.participant_pre_prompt)
            if mid_prompt:
                parts.append(mid_prompt)
            if self.participant_post_prompt:
                parts.append(self.participant_post_prompt)
            
            no_markdown_instruction = "\n\nIMPORTANT: Use plain text only. Do NOT use markdown formatting (no **bold**, no *italic*, no # headers, no lists with markdown syntax, etc.). Write in natural, plain text."
            
            if not parts:
                # Default prompt if none provided
                default_prompt = """You are participating in a moderated dialog. Another AI will moderate the conversation, and you will be in dialog with another participant. 
- Respond thoughtfully to questions and statements
- Build on previous points in the conversation
- Be clear and concise
- Engage constructively with the other participant's ideas"""
                return default_prompt + no_markdown_instruction
            
            return "\n\n".join(parts) + no_markdown_instruction

        # Set up system prompts
        self.intermediator.messages = [{"role": "system", "content": intermediator_system_prompt}]
        self.participant1.messages = [{"role": "system", "content": build_participant_prompt(self.participant1_mid_prompt)}]
        self.participant2.messages = [{"role": "system", "content": build_participant_prompt(self.participant2_mid_prompt)}]

        # Add context if provided
        if context_content:
            context_msg = f"""Here is context for our discussion:

{context_content}

Please use this context to inform your responses."""
            self.intermediator.messages.append({"role": "user", "content": context_msg})
            self.participant1.messages.append({"role": "user", "content": context_msg})
            self.participant2.messages.append({"role": "user", "content": context_msg})

        self._emit('dialog_started', {
            'intermediator': self.intermediator.name,
            'participant1': self.participant1.name,
            'participant2': self.participant2.name
        })

        # Preload all models in parallel to avoid delays on first turns
        self._emit('status_update', {
            'message': 'Preloading models on all servers (background)...'
        })

        preload_threads = []
        for client, name in [(self.intermediator, 'Intermediator'),
                             (self.participant1, 'Participant 1'),
                             (self.participant2, 'Participant 2')]:
            thread = threading.Thread(target=client.preload_model, daemon=True)
            thread.start()
            preload_threads.append(thread)

        # NOTE: We do NOT wait for preloads to complete, so the Intermediator can start immediately
        # for thread in preload_threads:
        #     thread.join(timeout=30)

        # self._emit('status_update', {
        #     'message': 'All models loaded. Starting dialog...'
        # })

        # Start with intermediator introducing the topic/starting the conversation
        # The topic is already in the system prompt, so just ask to begin
        intro_prompt = """Please introduce the topic to the two participants and start the conversation. Address both participants and set the stage for a productive discussion."""
        
        self._emit('intermediator_turn', {
            'turn': 0,
            'speaker': 'intermediator',
            'message': intro_prompt,
            'thinking': True,
            'intermediator': self.intermediator.name
        })

        intro_response, intro_tokens = self.intermediator.ask(intro_prompt, round_num=0)

        self.turn_counter = 0
        self.conversation_history.append({
            'turn': self.turn_counter,
            'speaker': 'intermediator',
            'message': intro_response,
            'tokens': intro_tokens,
            'thinking_enabled': self.intermediator.thinking
        })
        self.turn_counter += 1

        # Generate TTS for intro (in background thread)
        if self.enable_tts:
            tts_thread = threading.Thread(
                target=generate_tts_audio,
                args=(intro_response, 'intermediator', self.dialog_id,
                      self.intermediator_topic_prompt or 'Dialog', self.audio_sequence),
                daemon=True
            )
            tts_thread.start()
            self.tts_threads.append(tts_thread)
            self.audio_sequence += 1

        # Continue dialog for max_turns * 2 (so each participant goes max_turns times)
        for turn in range(1, (max_turns * 2) + 1):
            self.current_turn = turn
            
            # Build conversation context for participants
            recent_messages = self.conversation_history[-3:] if len(self.conversation_history) >= 3 else self.conversation_history
            context_summary = "\n\n".join([
                f"{msg['speaker']}: {msg['message'][:300]}"
                for msg in recent_messages
            ])
            
            # Alternate between participants
            if turn % 2 == 1:
                # Participant 1's turn
                # Get the last message from participant 2 or intermediator
                last_message = ""
                last_speaker = ""
                for msg in reversed(self.conversation_history):
                    if msg['speaker'] in ['participant2', 'intermediator']:
                        last_message = msg['message']
                        last_speaker = msg['speaker']
                        break
                
                p1_prompt = f"""Here's what has been said so far in our discussion:

{context_summary}

The {last_speaker} just said: "{last_message[:400]}"

Please respond thoughtfully. Engage with the points raised and contribute your perspective."""
                
                self._emit('participant_turn', {
                    'turn': turn,
                    'speaker': 'participant1',
                    'message': p1_prompt,
                    'thinking': True,
                    'participant1': self.participant1.name
                })

                p1_response, p1_tokens = self.participant1.ask(p1_prompt, round_num=turn)

                self.conversation_history.append({
                    'turn': self.turn_counter,
                    'speaker': 'participant1',
                    'message': p1_response,
                    'tokens': p1_tokens,
                    'thinking_enabled': self.participant1.thinking
                })
                self.turn_counter += 1

                # Generate TTS for participant 1 (in background thread)
                if self.enable_tts:
                    tts_thread = threading.Thread(
                        target=generate_tts_audio,
                        args=(p1_response, 'participant1', self.dialog_id,
                              self.intermediator_topic_prompt or 'Dialog', self.audio_sequence),
                        daemon=True
                    )
                    tts_thread.start()
                    self.tts_threads.append(tts_thread)
                    self.audio_sequence += 1

                # Update intermediator's context with participant 1's response
                self.intermediator.messages.append({
                    "role": "user",
                    "content": f"Participant 1 said: {p1_response}"
                })
                
                # Update participant 2's context so it sees participant 1's response
                self.participant2.messages.append({
                    "role": "user",
                    "content": f"Participant 1 said: {p1_response}"
                })

                # Intermediator moderates after each participant response to keep dialog on track
                # Skip moderation on the final turn since we go straight to summary
                if turn < (max_turns * 2):
                    mod_prompt = f"""Participant 1 just said: "{p1_response[:400]}"

You have access to the entire conversation history through your message context. As the moderator, please:
1. Ensure the conversation stays focused on the topic
2. If the conversation is drifting, gently redirect it back
3. If participants are making good progress, acknowledge it and encourage continuation
4. Ask clarifying questions if needed
5. Bridge connections between what different participants have said

Provide a brief moderation comment (2-3 sentences) that keeps the dialog productive and on track."""

                    self._emit('intermediator_turn', {
                        'turn': turn,
                        'speaker': 'intermediator',
                        'message': mod_prompt
                    })

                    mod_response, mod_tokens = self.intermediator.ask(mod_prompt, round_num=turn)

                    self.conversation_history.append({
                        'turn': self.turn_counter,
                        'speaker': 'intermediator',
                        'message': mod_response,
                        'tokens': mod_tokens,
                        'thinking_enabled': self.intermediator.thinking
                    })
                    self.turn_counter += 1

                    # Generate TTS for intermediator moderation (in background thread)
                    if self.enable_tts:
                        tts_thread = threading.Thread(
                            target=generate_tts_audio,
                            args=(mod_response, 'intermediator', self.dialog_id,
                                  self.intermediator_topic_prompt or 'Dialog', self.audio_sequence),
                            daemon=True
                        )
                        tts_thread.start()
                        self.tts_threads.append(tts_thread)
                        self.audio_sequence += 1

                    # Update participants' context
                    self.participant1.messages.append({
                        "role": "user",
                        "content": f"Moderator said: {mod_response}"
                    })
                    self.participant2.messages.append({
                        "role": "user",
                        "content": f"Moderator said: {mod_response}"
                    })

            else:
                # Participant 2's turn
                # Get the last message from participant 1 or intermediator
                last_message = ""
                last_speaker = ""
                for msg in reversed(self.conversation_history):
                    if msg['speaker'] in ['participant1', 'intermediator']:
                        last_message = msg['message']
                        last_speaker = msg['speaker']
                        break
                
                p2_prompt = f"""Here's what has been said so far in our discussion:

{context_summary}

The {last_speaker} just said: "{last_message[:400]}"

Please respond thoughtfully. Engage with the points raised and contribute your perspective."""
                
                self._emit('participant_turn', {
                    'turn': turn,
                    'speaker': 'participant2',
                    'message': p2_prompt,
                    'thinking': True,
                    'participant2': self.participant2.name
                })

                p2_response, p2_tokens = self.participant2.ask(p2_prompt, round_num=turn)

                self.conversation_history.append({
                    'turn': self.turn_counter,
                    'speaker': 'participant2',
                    'message': p2_response,
                    'tokens': p2_tokens,
                    'thinking_enabled': self.participant2.thinking
                })
                self.turn_counter += 1

                # Generate TTS for participant 2 (in background thread)
                if self.enable_tts:
                    tts_thread = threading.Thread(
                        target=generate_tts_audio,
                        args=(p2_response, 'participant2', self.dialog_id,
                              self.intermediator_topic_prompt or 'Dialog', self.audio_sequence),
                        daemon=True
                    )
                    tts_thread.start()
                    self.tts_threads.append(tts_thread)
                    self.audio_sequence += 1

                # Update intermediator's context with participant 2's response
                self.intermediator.messages.append({
                    "role": "user",
                    "content": f"Participant 2 said: {p2_response}"
                })
                
                # Update participant 1's context so it sees participant 2's response
                self.participant1.messages.append({
                    "role": "user",
                    "content": f"Participant 2 said: {p2_response}"
                })

                # Intermediator moderates after each participant response to keep dialog on track
                # Skip moderation on the final turn since we go straight to summary
                if turn < (max_turns * 2):
                    mod_prompt = f"""Participant 2 just said: "{p2_response[:400]}"

You have access to the entire conversation history through your message context. As the moderator, please:
1. Ensure the conversation stays focused on the topic
2. If the conversation is drifting, gently redirect it back
3. If participants are making good progress, acknowledge it and encourage continuation
4. Ask clarifying questions if needed
5. Bridge connections between what different participants have said

Provide a brief moderation comment (2-3 sentences) that keeps the dialog productive and on track."""

                    self._emit('intermediator_turn', {
                        'turn': turn,
                        'speaker': 'intermediator',
                        'message': mod_prompt
                    })

                    mod_response, mod_tokens = self.intermediator.ask(mod_prompt, round_num=turn)

                    self.conversation_history.append({
                        'turn': self.turn_counter,
                        'speaker': 'intermediator',
                        'message': mod_response,
                        'tokens': mod_tokens,
                        'thinking_enabled': self.intermediator.thinking
                    })
                    self.turn_counter += 1

                    # Generate TTS for intermediator moderation (in background thread)
                    if self.enable_tts:
                        tts_thread = threading.Thread(
                            target=generate_tts_audio,
                            args=(mod_response, 'intermediator', self.dialog_id,
                                  self.intermediator_topic_prompt or 'Dialog', self.audio_sequence),
                            daemon=True
                        )
                        tts_thread.start()
                        self.tts_threads.append(tts_thread)
                        self.audio_sequence += 1

                    # Update participants' context
                    self.participant1.messages.append({
                        "role": "user",
                        "content": f"Moderator said: {mod_response}"
                    })
                    self.participant2.messages.append({
                        "role": "user",
                        "content": f"Moderator said: {mod_response}"
                    })

        # Intermediator provides final summary
        summary_turn = self.turn_counter
        
        summary_prompt = """The dialog has now concluded. Please provide a comprehensive summary and wrap-up of the entire conversation. 

Based on all the exchanges between the participants and your moderation, please:

1. Summarize the main topic and key themes that were discussed
2. Review the key points, arguments, and perspectives raised by each participant
3. Identify areas of agreement or consensus that emerged
4. Identify areas of disagreement or different perspectives that were explored
5. Draw conclusions, insights, or takeaways that developed
6. Note any important questions or issues that remain unresolved or warrant further discussion

IMPORTANT: If this was a debate or competitive discussion where a winner can be determined, please:
- Evaluate the strength of each participant's arguments
- Assess the quality of their reasoning and evidence
- Consider how well they addressed counterarguments
- Declare a winner if you can determine one, explaining your reasoning
- If no clear winner can be determined, explain why (e.g., both sides made equally strong points, the discussion was exploratory rather than competitive, etc.)

Be thorough but concise. This summary should help anyone understand the essence of the dialog without reading every exchange, and if applicable, clearly indicate the outcome or winner of the debate."""
        
        self._emit('intermediator_turn', {
            'turn': summary_turn,
            'speaker': 'intermediator',
            'message': summary_prompt,
            'is_summary': True,
            'thinking': True,
            'intermediator': self.intermediator.name
        })

        summary_response, summary_tokens = self.intermediator.ask(summary_prompt, round_num=summary_turn)

        self.conversation_history.append({
            'turn': self.turn_counter,
            'speaker': 'intermediator',
            'message': summary_response,
            'tokens': summary_tokens,
            'is_summary': True,
            'thinking_enabled': self.intermediator.thinking
        })
        self.turn_counter += 1

        # Generate TTS for final summary (in background thread)
        if self.enable_tts:
            tts_thread = threading.Thread(
                target=generate_tts_audio,
                args=(summary_response, 'intermediator', self.dialog_id,
                      self.intermediator_topic_prompt or 'Dialog', self.audio_sequence),
                daemon=True
            )
            tts_thread.start()
            self.tts_threads.append(tts_thread)
            self.audio_sequence += 1

        # Share the summary with both participants
        self.participant1.messages.append({
            "role": "user",
            "content": f"Moderator's Final Summary: {summary_response}"
        })
        self.participant2.messages.append({
            "role": "user",
            "content": f"Moderator's Final Summary: {summary_response}"
        })

        # Wait for all TTS threads to complete before finishing (with timeout)
        for thread in self.tts_threads:
            if thread.is_alive():
                thread.join(timeout=30)  # Wait up to 30 seconds per thread

        self.end_time = time.time()
        runtime_seconds = self.end_time - self.start_time if self.start_time else 0

        self._emit('dialog_complete', {
            'conversation_history': self.conversation_history,
            'runtime_seconds': runtime_seconds,
            'total_turns': len(self.conversation_history)
        })
        
        # Return conversation data for saving
        return {
            'conversation_history': self.conversation_history,
            'runtime_seconds': runtime_seconds,
            'total_turns': len(self.conversation_history),
            'start_time': self.start_time,
            'end_time': self.end_time
        }


# Global state for web interface
app = Flask(__name__)
app.config['SECRET_KEY'] = 'intermediator-dialog-secret-key'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")

# Store uploaded files temporarily
uploaded_files = {}
file_usage_count = {}

# Store dialog instances
dialog_instances = {}

# Store dialog metadata
dialog_metadata = {}

# Store complete dialog data for PDF generation
complete_dialog_data = {}

# Store client instances to maintain conversation cache across dialogs
# Key format: "{host}:{model}:{name}"
client_instances = {}

# Store GPU monitoring data per dialog
# Key format: "{dialog_id}"
gpu_monitoring_data = {}

# Store GPU monitoring thread control
gpu_monitoring_threads = {}


def generate_tts_audio(text: str, speaker: str, dialog_id: str, topic: str, sequence: int) -> Optional[str]:
    """Generate TTS audio using OpenAI and save to file.

    Args:
        text: The text to convert to speech
        speaker: 'intermediator', 'participant1', or 'participant2'
        dialog_id: The dialog ID
        topic: The debate topic for folder naming
        sequence: Global sequence number for ordering

    Returns:
        Path to the saved audio file, or None if TTS fails
    """
    try:
        # Map speakers to OpenAI TTS voices
        voice_map = {
            'intermediator': 'alloy',    # Neutral, balanced voice
            'participant1': 'echo',       # Clear, articulate voice
            'participant2': 'fable'       # Warm, expressive voice
        }
        voice = voice_map.get(speaker, 'alloy')

        # Create folder name: Debate_{sanitized_topic}
        # Sanitize topic for folder name
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic)[:50]  # Remove special chars, limit length
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)  # Replace spaces/dashes with underscore
        folder_name = f"Debate_{sanitized_topic}" if sanitized_topic else f"Debate_{dialog_id[:8]}"

        # Create audio directory
        audio_dir = os.path.join('output', 'audio', folder_name)
        os.makedirs(audio_dir, exist_ok=True)

        # Generate filename with sequence number for proper ordering
        filename = f"{sequence:03d}_{speaker}.mp3"
        filepath = os.path.join(audio_dir, filename)

        # Emit TTS start event
        socketio.emit('tts_progress', {
            'status': 'generating',
            'speaker': speaker,
            'sequence': sequence,
            'filename': filename
        })

        # OpenAI TTS has a 4096 character limit - truncate if necessary
        max_length = 4000
        if len(text) > max_length:
            truncated_text = text[:max_length] + "..."
        else:
            truncated_text = text

        # Generate speech using OpenAI TTS
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=truncated_text
        )

        # Save audio file
        with open(filepath, 'wb') as f:
            f.write(response.content)

        # Emit TTS complete event
        socketio.emit('tts_progress', {
            'status': 'complete',
            'speaker': speaker,
            'sequence': sequence,
            'filename': filename,
            'filepath': filepath
        })

        return filepath

    except Exception as e:
        # Emit TTS error event
        socketio.emit('tts_progress', {
            'status': 'error',
            'speaker': speaker,
            'sequence': sequence,
            'error': str(e)
        })
        print(f"TTS Error for {speaker}: {str(e)}")
        return None


def generate_participant_summaries(dialog_data: Dict, participant1_client: 'OllamaClient',
                                  participant2_client: 'OllamaClient', dialog_id: str,
                                  topic: str, intermediator_config: Dict,
                                  participant1_config: Dict, participant2_config: Dict):
    """Save consolidated text files for each participant's turns.

    Args:
        dialog_data: The complete dialog data
        participant1_client: OllamaClient for participant A (unused, kept for compatibility)
        participant2_client: OllamaClient for participant B (unused, kept for compatibility)
        dialog_id: The dialog ID
        topic: The debate topic
        intermediator_config: Configuration for intermediator
        participant1_config: Configuration for participant A
        participant2_config: Configuration for participant B
    """
    try:
        conversation_history = dialog_data.get('conversation_history', [])

        # Aggregate turns for each participant
        participantA_turns = []
        participantB_turns = []

        for entry in conversation_history:
            speaker = entry.get('speaker')
            message = entry.get('message', '')
            if speaker == 'participant1':
                participantA_turns.append(message)
            elif speaker == 'participant2':
                participantB_turns.append(message)

        # Get participant names
        pA_name = participant1_config.get('name', 'Participant A')
        pB_name = participant2_config.get('name', 'Participant B')

        # Create sanitized topic for folder name
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic)[:50]
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)
        folder_name = f"Debate_{sanitized_topic}" if sanitized_topic else f"Debate_{dialog_id[:8]}"

        # Create summary directory
        audio_dir = os.path.join('output', 'audio', folder_name)
        os.makedirs(audio_dir, exist_ok=True)

        # Save consolidated text files with all turns
        pA_transcript_path = os.path.join(audio_dir, f"transcript_{pA_name.replace(' ', '_')}.txt")
        pB_transcript_path = os.path.join(audio_dir, f"transcript_{pB_name.replace(' ', '_')}.txt")

        with open(pA_transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"Participant A Transcript: {pA_name}\n")
            f.write(f"Topic: {topic}\n")
            f.write("=" * 80 + "\n\n")
            for i, turn in enumerate(participantA_turns, 1):
                f.write(f"Turn {i}:\n{turn}\n\n")

        with open(pB_transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"Participant B Transcript: {pB_name}\n")
            f.write(f"Topic: {topic}\n")
            f.write("=" * 80 + "\n\n")
            for i, turn in enumerate(participantB_turns, 1):
                f.write(f"Turn {i}:\n{turn}\n\n")

        debug_log('info', f"Saved participant transcripts to {audio_dir}")

        # Emit completion event (no diagrams)
        socketio.emit('summaries_generated', {
            'dialog_id': dialog_id,
            'participant1_summary': pA_transcript_path,
            'participant2_summary': pB_transcript_path,
            'participant1_diagram': None,
            'participant2_diagram': None
        })

        debug_log('info', "Participant transcripts saved successfully")

    except Exception as e:
        error_msg = f"Failed to save participant transcripts: {str(e)}"
        debug_log('error', error_msg)
        
        # Emit error event so frontend doesn't hang
        socketio.emit('summaries_error', {
            'dialog_id': dialog_id,
            'error': error_msg
        })


def debug_log(level, message, server=None, data=None):
    """Emit debug log event via WebSocket and also print to console."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message,
        'server': server
    }
    if data:
        log_entry['data'] = data

    socketio.emit('debug_log', log_entry)

    server_prefix = f"[{server}] " if server else ""
    print(f"[{timestamp}] [{level.upper()}] {server_prefix}{message}", flush=True)


def fetch_gpu_status(host: str) -> Optional[Dict]:
    """Fetch GPU status from a server's monitoring endpoint.

    Args:
        host: The host URL (e.g., 'http://192.168.5.40:11434')

    Returns:
        Dictionary with GPU status data or None if unavailable
    """
    try:
        # Extract base host without port 11434, use port 9999 for GPU monitoring
        base_host = host.split(':11434')[0]
        gpu_status_url = f"{base_host}:9999/gpu-status"

        response = requests.get(gpu_status_url, timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        # Silently fail - GPU monitoring is optional
        return None


def gpu_monitoring_thread(dialog_id: str, server_configs: Dict, poll_interval: float = 1.0):
    """Background thread to poll GPU status during dialog execution.

    Args:
        dialog_id: The dialog ID to track
        server_configs: Dictionary with 'intermediator', 'participant1', 'participant2' host configs
        poll_interval: How often to poll in seconds
    """
    if dialog_id not in gpu_monitoring_data:
        gpu_monitoring_data[dialog_id] = {
            'start_time': time.time(),
            'samples': [],
            'server_configs': server_configs
        }

    # Flag to control thread execution
    gpu_monitoring_threads[dialog_id] = {'running': True}

    while gpu_monitoring_threads[dialog_id].get('running', False):
        timestamp = time.time()
        sample = {
            'timestamp': timestamp,
            'elapsed': timestamp - gpu_monitoring_data[dialog_id]['start_time'],
            'servers': {}
        }

        # Poll each server
        for role, config in server_configs.items():
            if config and 'host' in config:
                gpu_data = fetch_gpu_status(config['host'])
                if gpu_data:
                    sample['servers'][role] = {
                        'hostname': gpu_data.get('hostname'),
                        'gpus': gpu_data.get('gpus', [])
                    }

        # Store sample
        gpu_monitoring_data[dialog_id]['samples'].append(sample)

        # Emit to frontend
        socketio.emit('gpu_status_update', {
            'dialog_id': dialog_id,
            'sample': sample
        })

        # Sleep until next poll
        time.sleep(poll_interval)

    # Mark end time when monitoring stops
    gpu_monitoring_data[dialog_id]['end_time'] = time.time()


def start_gpu_monitoring(dialog_id: str, intermediator_config: Dict,
                         participant1_config: Dict, participant2_config: Dict):
    """Start GPU monitoring for a dialog.

    Args:
        dialog_id: The dialog ID
        intermediator_config: Intermediator server configuration
        participant1_config: Participant 1 server configuration
        participant2_config: Participant 2 server configuration
    """
    server_configs = {
        'intermediator': intermediator_config,
        'participant1': participant1_config,
        'participant2': participant2_config
    }

    thread = threading.Thread(
        target=gpu_monitoring_thread,
        args=(dialog_id, server_configs),
        daemon=True
    )
    thread.start()
    debug_log('info', f"Started GPU monitoring for dialog {dialog_id}")


def stop_gpu_monitoring(dialog_id: str):
    """Stop GPU monitoring for a dialog.

    Args:
        dialog_id: The dialog ID
    """
    if dialog_id in gpu_monitoring_threads:
        gpu_monitoring_threads[dialog_id]['running'] = False
        debug_log('info', f"Stopped GPU monitoring for dialog {dialog_id}")


def generate_filename_from_topic(topic_prompt: str, max_length: int = 60) -> str:
    """Generate a unique, readable filename from the topic prompt."""
    if not topic_prompt:
        return f"dialog_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Extract first sentence or first 100 characters
    first_line = topic_prompt.split('\n')[0].strip()
    if len(first_line) > 100:
        first_line = first_line[:100]
    
    # Remove special characters and replace spaces with underscores
    filename = re.sub(r'[^\w\s-]', '', first_line)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.strip('_')
    
    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length].rstrip('_')
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # If filename is empty or too short, use timestamp-based name
    if not filename or len(filename) < 5:
        return f"dialog_{timestamp}"
    
    return f"{filename}_{timestamp}"


def save_dialog_to_files(dialog_data: Dict, prompt_config: Dict, 
                         intermediator_config: Dict, participant1_config: Dict, 
                         participant2_config: Dict, dialog_id: str):
    """Save dialog to JSON and TXT files.
    
    Returns:
        Tuple of (json_path, txt_path) or (None, None) on error
    """
    try:
        # Validate required inputs
        if not dialog_data:
            raise ValueError("dialog_data is required but was None or empty")
        if not prompt_config:
            raise ValueError("prompt_config is required but was None or empty")
        if not intermediator_config:
            raise ValueError("intermediator_config is required but was None or empty")
        if not participant1_config:
            raise ValueError("participant1_config is required but was None or empty")
        if not participant2_config:
            raise ValueError("participant2_config is required but was None or empty")
        if not dialog_id:
            raise ValueError("dialog_id is required but was None or empty")
        
        # Validate dialog_data structure
        if 'conversation_history' not in dialog_data:
            raise ValueError("dialog_data must contain 'conversation_history'")
        
        # Ensure output directory exists
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename from topic prompt
        topic_prompt = prompt_config.get('intermediator_topic_prompt', '')
        base_filename = generate_filename_from_topic(topic_prompt)
        
        # Prepare full dialog data for JSON
        json_data = {
            'dialog_id': dialog_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'intermediator': {
                    'host': intermediator_config.get('host'),
                    'model': intermediator_config.get('model'),
                    'name': intermediator_config.get('name')
                },
                'participant1': {
                    'host': participant1_config.get('host'),
                    'model': participant1_config.get('model'),
                    'name': participant1_config.get('name')
                },
                'participant2': {
                    'host': participant2_config.get('host'),
                    'model': participant2_config.get('model'),
                    'name': participant2_config.get('name')
                },
                'prompt_config': prompt_config,
                'runtime_seconds': dialog_data.get('runtime_seconds', 0),
                'total_turns': dialog_data.get('total_turns', 0),
                'start_time': datetime.fromtimestamp(dialog_data.get('start_time', time.time())).isoformat() if dialog_data.get('start_time') else None,
                'end_time': datetime.fromtimestamp(dialog_data.get('end_time', time.time())).isoformat() if dialog_data.get('end_time') else None
            },
            'conversation_history': dialog_data.get('conversation_history', [])
        }
        
        # Save JSON file
        json_path = output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Generate readable TXT file
        txt_path = output_dir / f"{base_filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("INTERMEDIATED DIALOG\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dialog ID: {dialog_id}\n")
            f.write(f"Timestamp: {json_data['timestamp']}\n")
            if dialog_data.get('start_time') and dialog_data.get('end_time'):
                f.write(f"Duration: {dialog_data.get('runtime_seconds', 0):.2f} seconds\n")
            f.write(f"Total Turns: {dialog_data.get('total_turns', 0)}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PARTICIPANTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Intermediator: {intermediator_config.get('name')} ({intermediator_config.get('model')})\n")
            f.write(f"Participant 1: {participant1_config.get('name')} ({participant1_config.get('model')})\n")
            f.write(f"Participant 2: {participant2_config.get('name')} ({participant2_config.get('model')})\n\n")
            
            if topic_prompt:
                f.write("-" * 80 + "\n")
                f.write("TOPIC / INSTRUCTIONS\n")
                f.write("-" * 80 + "\n")
                f.write(f"{topic_prompt}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("CONVERSATION\n")
            f.write("=" * 80 + "\n\n")
            
            # Write conversation history
            for entry in dialog_data.get('conversation_history', []):
                turn = entry.get('turn', 0)
                speaker = entry.get('speaker', 'unknown')
                message = entry.get('message', '')
                is_summary = entry.get('is_summary', False)
                tokens = entry.get('tokens', {})
                
                # Format speaker name
                if speaker == 'intermediator':
                    speaker_display = f"Moderator ({intermediator_config.get('name', 'Intermediator')})"
                elif speaker == 'participant1':
                    speaker_display = f"Participant 1 ({participant1_config.get('name', 'Participant 1')})"
                elif speaker == 'participant2':
                    speaker_display = f"Participant 2 ({participant2_config.get('name', 'Participant 2')})"
                else:
                    speaker_display = speaker.title()
                
                if is_summary:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("FINAL SUMMARY\n")
                    f.write("=" * 80 + "\n\n")
                else:
                    f.write(f"\n[Turn {turn}] {speaker_display}\n")
                    f.write("-" * 80 + "\n")
                
                f.write(f"{message}\n")
                
                if tokens and not is_summary:
                    total_tokens = tokens.get('total', 0)
                    if total_tokens > 0:
                        f.write(f"\n[Tokens: {total_tokens}]\n")
                
                f.write("\n")
        
        debug_log('info', f"Dialog saved to {json_path} and {txt_path}")
        return str(json_path), str(txt_path)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        debug_log('error', f"Failed to save dialog files: {str(e)}")
        debug_log('error', f"Error details: {error_details}")
        print(f"ERROR in save_dialog_to_files: {str(e)}", flush=True)
        print(f"Traceback: {error_details}", flush=True)
        return None, None


def clean_text_for_pdf(text: str) -> str:
    """Clean text for PDF generation by removing markdown and fixing encoding issues."""
    if not text:
        return text
    
    # Remove markdown formatting
    # Remove code blocks first (```code```) - do this before other processing
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove inline code (`code`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # Remove strikethrough (~~text~~)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    
    # Remove headers (# Header) - must be at start of line
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Remove bold (**text** or __text__) - handle properly matched pairs first
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Remove italic (*text* or _text_) - single asterisks/underscores
    # Be careful to match pairs, but handle edge cases
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_([^_\n]+?)_(?!_)', r'\1', text)
    
    # Remove any remaining unmatched markdown asterisks/underscores
    # This handles cases like "*text** or malformed markdown
    # Only remove if they appear to be markdown (not standalone punctuation)
    text = re.sub(r'\*\*+', '', text)  # Remove sequences of 2+ asterisks
    text = re.sub(r'__+', '', text)    # Remove sequences of 2+ underscores
    # Remove single asterisks/underscores that appear to be markdown (surrounded by word chars)
    text = re.sub(r'(?<=\w)\*(?=\w)', '', text)  # Asterisk between word chars
    text = re.sub(r'(?<=\w)_(?=\w)', '', text)   # Underscore between word chars
    
    # Fix character encoding issues - replace problematic characters
    # Replace various dash characters with regular hyphens
    text = text.replace('\u2010', '-')  # Hyphen
    text = text.replace('\u2011', '-')  # Non-breaking hyphen
    text = text.replace('\u2012', '-')  # Figure dash
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2015', '-')  # Horizontal bar
    text = text.replace('\u2212', '-')  # Minus sign
    text = text.replace('\u25A0', ' ')  # Black square (■)

    # Replace various space-like characters with regular spaces
    text = text.replace('\u00A0', ' ')  # Non-breaking space
    text = text.replace('\u2000', ' ')  # En quad
    text = text.replace('\u2001', ' ')  # Em quad
    text = text.replace('\u2002', ' ')  # En space
    text = text.replace('\u2003', ' ')  # Em space
    text = text.replace('\u2004', ' ')  # Three-per-em space
    text = text.replace('\u2005', ' ')  # Four-per-em space
    text = text.replace('\u2006', ' ')  # Six-per-em space
    text = text.replace('\u2007', ' ')  # Figure space
    text = text.replace('\u2008', ' ')  # Punctuation space
    text = text.replace('\u2009', ' ')  # Thin space
    text = text.replace('\u200A', ' ')  # Hair space
    
    # Normalize multiple spaces to single space (but preserve line breaks)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Clean up any remaining markdown artifacts
    # Handle malformed patterns like "*text** or "*text*text**
    text = re.sub(r'\*+([^*\n]+?)\*+', r'\1', text)  # Remove any asterisks around text
    text = re.sub(r'_+([^_\n]+?)_+', r'\1', text)    # Remove any underscores around text
    
    # Remove isolated markdown characters
    text = re.sub(r'\s+\*\s+', ' ', text)  # Remove isolated asterisks with spaces
    text = re.sub(r'\s+_\s+', ' ', text)   # Remove isolated underscores with spaces
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)  # Remove leading asterisks
    text = re.sub(r'\s*\*+$', '', text, flags=re.MULTILINE)  # Remove trailing asterisks
    
    return text.strip()


def generate_pdf_from_dialog(dialog_data: Dict, prompt_config: Dict,
                             intermediator_config: Dict, participant1_config: Dict,
                             participant2_config: Dict, dialog_id: str) -> Optional[str]:
    """Generate a PDF from dialog data with metadata, timing, and token counts.
    
    Returns:
        Path to generated PDF file, or None on error
    """
    try:
        # Validate required inputs
        if not dialog_data:
            raise ValueError("dialog_data is required but was None or empty")
        if not prompt_config:
            raise ValueError("prompt_config is required but was None or empty")
        if not intermediator_config:
            raise ValueError("intermediator_config is required but was None or empty")
        if not participant1_config:
            raise ValueError("participant1_config is required but was None or empty")
        if not participant2_config:
            raise ValueError("participant2_config is required but was None or empty")
        if not dialog_id:
            raise ValueError("dialog_id is required but was None or empty")
        
        # Validate dialog_data structure
        if 'conversation_history' not in dialog_data:
            raise ValueError("dialog_data must contain 'conversation_history'")
        
        # Ensure output directory exists
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename from topic prompt
        topic_prompt = prompt_config.get('intermediator_topic_prompt', '')
        base_filename = generate_filename_from_topic(topic_prompt)
        pdf_path = output_dir / f"{base_filename}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1f4e78'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f4e78'),
            spaceAfter=8,
            spaceBefore=12
        )
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        normal_style.leading = 14
        
        # Title
        elements.append(Paragraph("Intermediated Dialog Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata Section
        elements.append(Paragraph("Metadata", heading_style))
        
        # Dialog ID and Timestamp
        metadata_table_data = [
            ['Dialog ID:', dialog_id],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        if dialog_data.get('start_time') and dialog_data.get('end_time'):
            start_dt = datetime.fromtimestamp(dialog_data.get('start_time'))
            end_dt = datetime.fromtimestamp(dialog_data.get('end_time'))
            metadata_table_data.extend([
                ['Start Time:', start_dt.strftime('%Y-%m-%d %H:%M:%S')],
                ['End Time:', end_dt.strftime('%Y-%m-%d %H:%M:%S')],
                ['Duration:', f"{dialog_data.get('runtime_seconds', 0):.2f} seconds ({dialog_data.get('runtime_seconds', 0)/60:.2f} minutes)"],
            ])
        
        metadata_table_data.extend([
            ['Total Turns:', str(dialog_data.get('total_turns', 0))],
        ])
        
        metadata_table = Table(metadata_table_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e7f3ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(metadata_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Participants Section
        elements.append(Paragraph("Participants", heading_style))
        
        participants_data = [
            ['Role', 'Name', 'Host', 'Model'],
            ['Intermediator', intermediator_config.get('name', 'N/A'),
             intermediator_config.get('host', 'N/A'), intermediator_config.get('model', 'N/A')],
            ['Participant 1', participant1_config.get('name', 'N/A'),
             participant1_config.get('host', 'N/A'), participant1_config.get('model', 'N/A')],
            ['Participant 2', participant2_config.get('name', 'N/A'),
             participant2_config.get('host', 'N/A'), participant2_config.get('model', 'N/A')],
        ]
        
        participants_table = Table(participants_data, colWidths=[1.5*inch, 2*inch, 2*inch, 2*inch])
        participants_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#dae3f3')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#e2efda')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#fce4d6')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(participants_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add Energy Estimate if available
        # Access global complete_dialog_data to get GPU data
        gpu_data = {}
        if dialog_id in complete_dialog_data:
            gpu_data = complete_dialog_data[dialog_id].get('gpu_data', {})

        if gpu_data and 'samples' in gpu_data:
            total_energy = 0.0

            # Calculate energy from GPU samples
            # GPU data structure: {'start_time': ..., 'samples': [...], 'server_configs': ...}
            samples = gpu_data.get('samples', [])

            # Track energy per server role
            server_energy = {
                'intermediator': 0.0,
                'participant1': 0.0,
                'participant2': 0.0
            }

            # Time interval between samples (default 1 second)
            interval_hours = 1.0 / 3600.0

            for sample in samples:
                servers = sample.get('servers', {})
                for role in ['intermediator', 'participant1', 'participant2']:
                    server_data = servers.get(role, {})
                    gpus = server_data.get('gpus', [])
                    for gpu in gpus:
                        power_draw = gpu.get('power_draw_watts', 0)
                        if power_draw > 0:
                            server_energy[role] += power_draw * interval_hours

            # Sum total energy
            total_energy = sum(server_energy.values())

            if total_energy > 0:
                elements.append(Paragraph("Estimated Energy Consumption", heading_style))

                # Show breakdown by server
                energy_text = f"Total Energy: {total_energy:.4f} Wh<br/>"
                for role, energy in server_energy.items():
                    if energy > 0:
                        role_name = role.replace('_', ' ').title()
                        energy_text += f"{role_name}: {energy:.4f} Wh<br/>"

                elements.append(Paragraph(energy_text, normal_style))
                elements.append(Spacer(1, 0.2*inch))
        
        # Prompt Configuration Section
        if topic_prompt:
            elements.append(Paragraph("Topic / Instructions", heading_style))
            cleaned_topic = clean_text_for_pdf(topic_prompt)
            elements.append(Paragraph(cleaned_topic.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        if prompt_config.get('intermediator_pre_prompt'):
            elements.append(Paragraph("Intermediator Pre-Prompt", heading_style))
            cleaned_pre = clean_text_for_pdf(prompt_config.get('intermediator_pre_prompt'))
            elements.append(Paragraph(cleaned_pre.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        if prompt_config.get('participant_pre_prompt'):
            elements.append(Paragraph("Participant Pre-Prompt (Shared)", heading_style))
            cleaned_pre = clean_text_for_pdf(prompt_config.get('participant_pre_prompt'))
            elements.append(Paragraph(cleaned_pre.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        if prompt_config.get('participant1_mid_prompt'):
            elements.append(Paragraph("Participant 1 Mid-Prompt", heading_style))
            cleaned_mid = clean_text_for_pdf(prompt_config.get('participant1_mid_prompt'))
            elements.append(Paragraph(cleaned_mid.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        if prompt_config.get('participant2_mid_prompt'):
            elements.append(Paragraph("Participant 2 Mid-Prompt", heading_style))
            cleaned_mid = clean_text_for_pdf(prompt_config.get('participant2_mid_prompt'))
            elements.append(Paragraph(cleaned_mid.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        if prompt_config.get('participant_post_prompt'):
            elements.append(Paragraph("Participant Post-Prompt (Shared)", heading_style))
            cleaned_post = clean_text_for_pdf(prompt_config.get('participant_post_prompt'))
            elements.append(Paragraph(cleaned_post.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(PageBreak())
        
        # Conversation Section
        elements.append(Paragraph("Full Dialog", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        conversation_history = dialog_data.get('conversation_history', [])
        
        # Calculate totals (exclude entries with thinking enabled)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        thinking_turns_count = 0
        
        for entry in conversation_history:
            tokens = entry.get('tokens', {})
            thinking_enabled = entry.get('thinking_enabled', False)
            if tokens and not thinking_enabled:
                total_prompt_tokens += tokens.get('prompt_tokens', 0)
                total_completion_tokens += tokens.get('completion_tokens', 0)
                total_tokens += tokens.get('total', 0)
            elif thinking_enabled:
                thinking_turns_count += 1
        
        # Write each turn
        for idx, entry in enumerate(conversation_history):
            turn = entry.get('turn', idx)
            speaker = entry.get('speaker', 'unknown')
            message = entry.get('message', '')
            is_summary = entry.get('is_summary', False)
            tokens = entry.get('tokens', {})
            
            # Format speaker name
            if speaker == 'intermediator':
                speaker_display = f"Moderator ({intermediator_config.get('name', 'Intermediator')})"
                bg_color = colors.HexColor('#dae3f3')
            elif speaker == 'participant1':
                speaker_display = f"Participant 1 ({participant1_config.get('name', 'Participant 1')})"
                bg_color = colors.HexColor('#e2efda')
            elif speaker == 'participant2':
                speaker_display = f"Participant 2 ({participant2_config.get('name', 'Participant 2')})"
                bg_color = colors.HexColor('#fce4d6')
            else:
                speaker_display = speaker.title()
                bg_color = colors.white
            
            # Turn header
            if is_summary:
                turn_label = "FINAL SUMMARY"
                header_style = ParagraphStyle(
                    'SummaryHeader',
                    parent=styles['Heading2'],
                    fontSize=14,
                    textColor=colors.HexColor('#1f4e78'),
                    spaceAfter=8,
                    spaceBefore=12,
                    alignment=TA_CENTER
                )
                elements.append(Paragraph(turn_label, header_style))
            else:
                turn_label = f"Turn {turn}: {speaker_display}"
                header_style = ParagraphStyle(
                    'TurnHeader',
                    parent=styles['Heading3'],
                    fontSize=12,
                    textColor=colors.HexColor('#1f4e78'),
                    spaceAfter=6,
                    spaceBefore=10
                )
                elements.append(Paragraph(turn_label, header_style))

            # Add white space before colored box
            elements.append(Spacer(1, 0.12*inch))

            # Message content
            message_style = ParagraphStyle(
                'Message',
                parent=normal_style,
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY,
                leftIndent=0.2*inch,
                rightIndent=0.2*inch,
                backColor=bg_color,
                borderPadding=10
            )
            # Clean text (remove markdown, fix encoding) then escape HTML and preserve line breaks
            cleaned_message = clean_text_for_pdf(message)
            message_escaped = cleaned_message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
            elements.append(Paragraph(message_escaped, message_style))

            # Add white space after colored box
            elements.append(Spacer(1, 0.12*inch))

            # Token information (exclude if thinking mode was enabled, as tokens include thinking)
            thinking_enabled = entry.get('thinking_enabled', False)
            if tokens and not is_summary and not thinking_enabled:
                token_info_parts = []
                if tokens.get('prompt_tokens', 0) > 0:
                    token_info_parts.append(f"Prompt: {tokens.get('prompt_tokens', 0)}")
                if tokens.get('completion_tokens', 0) > 0:
                    token_info_parts.append(f"Completion: {tokens.get('completion_tokens', 0)}")
                if tokens.get('total', 0) > 0:
                    token_info_parts.append(f"Total: {tokens.get('total', 0)}")
                if tokens.get('tokens_per_second', 0) > 0:
                    token_info_parts.append(f"Speed: {tokens.get('tokens_per_second', 0):.2f} tokens/sec")
                if tokens.get('time_to_first_token', 0) > 0:
                    token_info_parts.append(f"TTFT: {tokens.get('time_to_first_token', 0):.3f}s")

                if token_info_parts:
                    token_text = " | ".join(token_info_parts)
                    token_style = ParagraphStyle(
                        'TokenInfo',
                        parent=normal_style,
                        fontSize=9,
                        textColor=colors.grey,
                        fontStyle='italic',
                        spaceBefore=4
                    )
                    elements.append(Paragraph(f"<i>Tokens: {token_text}</i>", token_style))
            elif thinking_enabled and tokens:
                # Show a note that tokens are not shown because thinking mode includes thinking tokens
                token_style = ParagraphStyle(
                    'TokenInfo',
                    parent=normal_style,
                    fontSize=9,
                    textColor=colors.grey,
                    fontStyle='italic',
                    spaceBefore=4
                )
                elements.append(Paragraph("<i>Tokens: Not shown (thinking mode enabled - tokens include thinking process)</i>", token_style))

            # Always add spacing between turns
            elements.append(Spacer(1, 0.05*inch))
        
        # Summary statistics
        elements.append(PageBreak())
        elements.append(Paragraph("Summary Statistics", heading_style))
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Turns', str(len(conversation_history))],
            ['Total Prompt Tokens', f"{total_prompt_tokens:,}"],
            ['Total Completion Tokens', f"{total_completion_tokens:,}"],
            ['Total Tokens', f"{total_tokens:,}"],
        ]
        
        if thinking_turns_count > 0:
            stats_data.append(['Turns with Thinking Mode', f"{thinking_turns_count} (tokens excluded from counts)"])
        
        if dialog_data.get('runtime_seconds', 0) > 0 and total_tokens > 0:
            avg_tokens_per_second = total_tokens / dialog_data.get('runtime_seconds', 1)
            stats_data.append(['Average Tokens/Second', f"{avg_tokens_per_second:.2f}"])
        
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ]))
        elements.append(stats_table)
        
        # Build PDF
        doc.build(elements)
        
        # Verify PDF was created
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file was not created at {pdf_path}")
        
        debug_log('info', f"PDF generated: {pdf_path}")
        return str(pdf_path)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        debug_log('error', f"Failed to generate PDF: {str(e)}")
        debug_log('error', f"Error details: {error_details}")
        print(f"ERROR in generate_pdf_from_dialog: {str(e)}", flush=True)
        print(f"Traceback: {error_details}", flush=True)
        traceback.print_exc()
        return None


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('intermediator_dialog.html')


@app.route('/debate_library', methods=['GET'])
def get_debate_library():
    """Serve the debate library JSON."""
    try:
        library_path = Path(__file__).parent / 'debate_library.json'
        if not library_path.exists():
            return jsonify({'error': 'Debate library not found'}), 404

        with open(library_path, 'r', encoding='utf-8') as f:
            library_data = json.load(f)

        return jsonify(library_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_debate', methods=['POST'])
def save_debate():
    """Save a new debate to the library."""
    try:
        debate_data = request.json

        # Validate required fields
        required_fields = ['id', 'name', 'description', 'intermediator_topic_prompt']
        for field in required_fields:
            if not debate_data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate ID format
        debate_id = debate_data['id']
        if not re.match(r'^[a-z0-9_]+$', debate_id):
            return jsonify({'error': 'ID must contain only lowercase letters, numbers, and underscores'}), 400

        library_path = Path(__file__).parent / 'debate_library.json'

        # Load existing library
        if library_path.exists():
            with open(library_path, 'r', encoding='utf-8') as f:
                library = json.load(f)
        else:
            library = {'debates': []}

        # Check if ID already exists
        existing_ids = [d['id'] for d in library['debates']]
        if debate_id in existing_ids:
            return jsonify({'error': f'A debate with ID "{debate_id}" already exists'}), 400

        # Create new debate entry
        new_debate = {
            'id': debate_data['id'],
            'name': debate_data['name'],
            'description': debate_data['description'],
            'intermediator_pre_prompt': debate_data.get('intermediator_pre_prompt', ''),
            'intermediator_topic_prompt': debate_data['intermediator_topic_prompt'],
            'participant_pre_prompt': debate_data.get('participant_pre_prompt', ''),
            'participant1_mid_prompt': debate_data.get('participant1_mid_prompt', ''),
            'participant2_mid_prompt': debate_data.get('participant2_mid_prompt', ''),
            'participant_post_prompt': debate_data.get('participant_post_prompt', ''),
            'max_turns': debate_data.get('max_turns', 4)
        }

        # Add to library
        library['debates'].append(new_debate)

        # Save back to file
        with open(library_path, 'w', encoding='utf-8') as f:
            json.dump(library, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': f'Debate "{new_debate["name"]}" saved successfully', 'debate_count': len(library['debates'])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_pdf/<dialog_id>', methods=['GET'])
def generate_pdf(dialog_id):
    """Generate and serve PDF for a completed dialog."""
    if dialog_id not in complete_dialog_data:
        return jsonify({'error': 'Dialog not found or not completed'}), 404

    data = complete_dialog_data[dialog_id]

    # Generate PDF
    pdf_path = generate_pdf_from_dialog(
        data['dialog_data'],
        data['prompt_config'],
        data['intermediator_config'],
        data['participant1_config'],
        data['participant2_config'],
        dialog_id
    )

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'Failed to generate PDF'}), 500

    # Send PDF file
    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=os.path.basename(pdf_path)
    )


@app.route('/reset_cache', methods=['POST'])
def reset_cache():
    """Reset conversation cache for specified clients."""
    data = request.get_json()
    reset_all = data.get('reset_all', False)
    client_keys = data.get('client_keys', [])
    
    if reset_all:
        # Reset all client caches
        for key, client in client_instances.items():
            client.reset_conversation()
        debug_log('info', f"Reset conversation cache for all clients ({len(client_instances)} clients)")
        return jsonify({'message': f'Reset cache for all {len(client_instances)} clients', 'reset_count': len(client_instances)})
    elif client_keys:
        # Reset specific clients
        reset_count = 0
        for key in client_keys:
            if key in client_instances:
                client_instances[key].reset_conversation()
                reset_count += 1
        debug_log('info', f"Reset conversation cache for {reset_count} specified clients")
        return jsonify({'message': f'Reset cache for {reset_count} clients', 'reset_count': reset_count})
    else:
        return jsonify({'error': 'No clients specified'}), 400


@app.route('/gpu_monitoring/<dialog_id>', methods=['GET'])
def get_gpu_monitoring_data(dialog_id):
    """Retrieve GPU monitoring data for a dialog."""
    if dialog_id not in gpu_monitoring_data:
        return jsonify({'error': 'GPU monitoring data not found for this dialog'}), 404

    data = gpu_monitoring_data[dialog_id]
    return jsonify({
        'dialog_id': dialog_id,
        'start_time': data.get('start_time'),
        'end_time': data.get('end_time'),
        'duration': data.get('end_time', time.time()) - data.get('start_time', 0),
        'sample_count': len(data.get('samples', [])),
        'samples': data.get('samples', []),
        'server_configs': data.get('server_configs', {})
    })


@socketio.on('reset_dialog_cache')
def handle_reset_cache(data):
    """Reset conversation cache for dialog participants."""
    reset_all = data.get('reset_all', False)
    intermediator_key = data.get('intermediator_key')
    participant1_key = data.get('participant1_key')
    participant2_key = data.get('participant2_key')
    
    reset_count = 0
    
    if reset_all:
        for key, client in client_instances.items():
            client.reset_conversation()
            reset_count += 1
        emit('cache_reset', {'message': f'Reset cache for all {reset_count} clients', 'reset_count': reset_count})
    else:
        keys_to_reset = []
        if intermediator_key and intermediator_key in client_instances:
            keys_to_reset.append(intermediator_key)
        if participant1_key and participant1_key in client_instances:
            keys_to_reset.append(participant1_key)
        if participant2_key and participant2_key in client_instances:
            keys_to_reset.append(participant2_key)
        
        for key in keys_to_reset:
            client_instances[key].reset_conversation()
            reset_count += 1
        
        emit('cache_reset', {'message': f'Reset cache for {reset_count} clients', 'reset_count': reset_count})


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

        client = OllamaClient(host, model, name)
        is_available = client.check_server_available()

        available_models = []
        if is_available:
            try:
                url = f"{host.rstrip('/')}/api/tags"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model.get('name', '') for model in models_data.get('models', [])]
            except:
                pass

        server_statuses.append({
            'name': name,
            'host': host,
            'model': model,
            'available': is_available,
            'available_models': available_models
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
    
    temp_path = None
    try:
        suffix = '_' + secure_filename(file.filename) if file.filename else ''
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            file.seek(0)
            tmp_file.write(file.read())
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise Exception('File was not saved correctly')
        
        session_id = str(uuid.uuid4())
        uploaded_files[session_id] = temp_path
        debug_log('info', f"File uploaded: {file.filename} -> {temp_path} (session_id: {session_id})")
        return jsonify({'session_id': session_id, 'filename': file.filename})
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500


def run_dialog_thread(intermediator_client, participant1_client, participant2_client,
                      context_file: str, dialog_id: str, max_turns: int = 3,
                      session_id: str = None, thinking_params: Dict = None,
                      prompt_config: Dict = None, intermediator_config: Dict = None,
                      participant1_config: Dict = None, participant2_config: Dict = None,
                      enable_tts: bool = True):
    """Run dialog in a separate thread."""
    try:
        # Apply thinking parameters if provided
        # Note: Thinking mode should only be applied to participants, not the intermediator
        # to avoid mixing reasoning with the actual moderation speech
        if thinking_params:
            if 'thinking' in thinking_params:
                # Only apply thinking to participants, not intermediator
                participant1_client.thinking = thinking_params['thinking']
                participant2_client.thinking = thinking_params['thinking']
            if 'temperature' in thinking_params:
                intermediator_client.temperature = thinking_params['temperature']
                participant1_client.temperature = thinking_params['temperature']
                participant2_client.temperature = thinking_params['temperature']
            # Add other parameters as needed

        # Extract prompt configuration
        prompt_config = prompt_config or {}
        dialog = IntermediatorDialog(
            intermediator_client, participant1_client, participant2_client,
            intermediator_pre_prompt=prompt_config.get('intermediator_pre_prompt'),
            intermediator_topic_prompt=prompt_config.get('intermediator_topic_prompt'),
            participant_pre_prompt=prompt_config.get('participant_pre_prompt'),
            participant1_mid_prompt=prompt_config.get('participant1_mid_prompt'),
            participant2_mid_prompt=prompt_config.get('participant2_mid_prompt'),
            participant_post_prompt=prompt_config.get('participant_post_prompt'),
            dialog_id=dialog_id,
            enable_tts=enable_tts
        )
        dialog_instances[dialog_id] = dialog

        def make_callback(role):
            def callback(event_data):
                event_data['role'] = role
                socketio.emit('dialog_update', event_data)
            return callback

        dialog.set_stream_callback(lambda data: socketio.emit('dialog_update', data))

        # Start GPU monitoring
        start_gpu_monitoring(dialog_id, intermediator_config, participant1_config, participant2_config)

        dialog_result = dialog.run_dialog(max_turns=max_turns, context_file=context_file)

        # Stop GPU monitoring
        stop_gpu_monitoring(dialog_id)
        
        # Save dialog to files
        if dialog_result and intermediator_config and participant1_config and participant2_config:
            # Validate dialog_result structure
            if not isinstance(dialog_result, dict):
                raise ValueError(f"dialog_result must be a dict, got {type(dialog_result)}")
            if 'conversation_history' not in dialog_result:
                raise ValueError("dialog_result must contain 'conversation_history'")
            
            json_path, txt_path = save_dialog_to_files(
                dialog_result, prompt_config, intermediator_config, 
                participant1_config, participant2_config, dialog_id
            )
            if json_path and txt_path:
                socketio.emit('dialog_saved', {
                    'dialog_id': dialog_id,
                    'json_path': json_path,
                    'txt_path': txt_path
                })
                debug_log('info', f"Dialog files saved successfully for {dialog_id}")
            else:
                debug_log('warning', f"Dialog file saving returned None - files may not have been saved for {dialog_id}")
                socketio.emit('error', {
                    'error': f'Failed to save dialog files for dialog {dialog_id}'
                })
            
            # Store complete dialog data for PDF generation (even if file save failed)
            # Get GPU data for this dialog
            gpu_data = gpu_monitoring_data.get(dialog_id, {})

            complete_dialog_data[dialog_id] = {
                'dialog_data': dialog_result,
                'prompt_config': prompt_config,
                'intermediator_config': intermediator_config,
                'participant1_config': participant1_config,
                'participant2_config': participant2_config,
                'gpu_data': gpu_data
            }

            # Emit event to notify frontend that PDF is ready
            socketio.emit('pdf_ready', {
                'dialog_id': dialog_id
            })
            debug_log('info', f"PDF data stored and ready for dialog {dialog_id}")

            # Generate participant summaries in background
            topic = prompt_config.get('intermediator_topic_prompt', 'Debate')
            summary_thread = threading.Thread(
                target=generate_participant_summaries,
                args=(dialog_result, participant1_client, participant2_client, dialog_id,
                      topic, intermediator_config, participant1_config, participant2_config),
                daemon=True
            )
            summary_thread.start()

    except Exception as e:
        error_msg = str(e)
        socketio.emit('error', {
            'error': error_msg
        })
        # Emit summaries_generated event even on error to ensure UI doesn't hang
        socketio.emit('summaries_generated', {
            'dialog_id': dialog_id,
            'summary_path': None,
            'diagram_path': None
        })
        # Don't emit pdf_ready on error since PDF generation would fail anyway
        debug_log('error', f"Dialog thread error: {error_msg}")
    finally:
        # Clean up uploaded file if needed
        if session_id and session_id in file_usage_count:
            file_usage_count[session_id] -= 1
            if file_usage_count[session_id] <= 0:
                if session_id in uploaded_files:
                    temp_path = uploaded_files.pop(session_id)
                    if os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                if session_id in file_usage_count:
                    del file_usage_count[session_id]


@socketio.on('start_dialog')
def handle_start_dialog(data):
    """Start a new intermediator dialog."""
    context_file = data.get('context_file')
    session_id = data.get('session_id')
    intermediator_config = data.get('intermediator')
    participant1_config = data.get('participant1')
    participant2_config = data.get('participant2')
    max_turns = data.get('max_turns', 3)
    thinking_params = data.get('thinking_params', {})
    prompt_config = data.get('prompt_config', {})
    enable_tts = data.get('enable_tts', True)

    # Validate that intermediator topic prompt is provided
    if not prompt_config.get('intermediator_topic_prompt'):
        emit('error', {'error': 'Intermediator topic/instructions prompt is required'})
        return
    
    # Handle shared model for Spark servers
    shared_spark_model = data.get('shared_spark_model')
    participant1_override_model = data.get('participant1_override_model')
    participant2_override_model = data.get('participant2_override_model')
    
    # Use shared model if provided and not overridden
    # If override_model is provided, use it; otherwise use shared model
    if shared_spark_model:
        if participant1_override_model:
            participant1_config['model'] = participant1_override_model
        else:
            participant1_config['model'] = shared_spark_model
        if participant2_override_model:
            participant2_config['model'] = participant2_override_model
        else:
            participant2_config['model'] = shared_spark_model

    if not intermediator_config or not participant1_config or not participant2_config:
        emit('error', {'error': 'All three AI configurations required'})
        return

    # Get uploaded file path if session_id provided
    if session_id and session_id in uploaded_files:
        context_file = uploaded_files[session_id]
        if context_file and os.path.exists(context_file):
            file_size = os.path.getsize(context_file)
            debug_log('info', f"Using uploaded file: session_id={session_id}, path={context_file}, size={file_size} bytes")
        else:
            context_file = None

    # Create dialog ID
    dialog_id = str(uuid.uuid4())
    
    # Store metadata
    dialog_metadata[dialog_id] = {
        'intermediator_topic_prompt': prompt_config.get('intermediator_topic_prompt', '')[:200] + '...' if prompt_config.get('intermediator_topic_prompt') else None,
        'context_filename': os.path.basename(context_file) if context_file else None,
        'timestamp': datetime.now().isoformat(),
        'intermediator': intermediator_config.get('name'),
        'participant1': participant1_config.get('name'),
        'participant2': participant2_config.get('name'),
        'max_turns': max_turns
    }

    # Create or reuse clients (to maintain conversation cache)
    def get_or_create_client(config, role_name):
        """Get existing client instance or create new one to maintain cache."""
        key = f"{config['host']}:{config['model']}:{config.get('name', role_name)}"
        
        if key in client_instances:
            # Reuse existing client to maintain conversation cache
            client = client_instances[key]
            # Update parameters if provided
            if thinking_params.get('num_ctx'):
                client.num_ctx = thinking_params.get('num_ctx', 8192)
            if thinking_params.get('temperature') is not None:
                client.temperature = thinking_params.get('temperature')
            if thinking_params.get('top_p') is not None:
                client.top_p = thinking_params.get('top_p')
            if thinking_params.get('top_k') is not None:
                client.top_k = thinking_params.get('top_k')
            if thinking_params.get('repeat_penalty') is not None:
                client.repeat_penalty = thinking_params.get('repeat_penalty')
            if thinking_params.get('num_predict') is not None:
                client.num_predict = thinking_params.get('num_predict')
            if 'thinking' in thinking_params:
                client.thinking = thinking_params.get('thinking', False)
            if 'be_brief' in thinking_params:
                client.be_brief = thinking_params.get('be_brief', False)
            debug_log('info', f"Reusing cached client for {role_name}: {key}", server=role_name)
            return client
        else:
            # Create new client
            client = OllamaClient(
                config['host'],
                config['model'],
                config.get('name'),
                num_ctx=thinking_params.get('num_ctx', 8192),
                temperature=thinking_params.get('temperature'),
                top_p=thinking_params.get('top_p'),
                top_k=thinking_params.get('top_k'),
                repeat_penalty=thinking_params.get('repeat_penalty'),
                num_predict=thinking_params.get('num_predict'),
                thinking=thinking_params.get('thinking', False),
                be_brief=thinking_params.get('be_brief', False)
            )
            client_instances[key] = client
            debug_log('info', f"Created new client for {role_name}: {key}", server=role_name)
            return client

    intermediator_client = get_or_create_client(intermediator_config, 'intermediator')
    participant1_client = get_or_create_client(participant1_config, 'participant1')
    participant2_client = get_or_create_client(participant2_config, 'participant2')

    # Check server availability
    if not intermediator_client.check_server_available():
        emit('error', {'error': f'Intermediator server {intermediator_config["host"]} not available'})
        return
    if not participant1_client.check_server_available():
        emit('error', {'error': f'Participant 1 server {participant1_config["host"]} not available'})
        return
    if not participant2_client.check_server_available():
        emit('error', {'error': f'Participant 2 server {participant2_config["host"]} not available'})
        return

    # Track file usage
    if session_id and session_id in uploaded_files:
        file_usage_count[session_id] = 1

    # Emit dialog started
    emit('dialog_started', {
        'dialog_id': dialog_id,
        'has_context_file': bool(context_file),
        'context_filename': os.path.basename(context_file) if context_file else None,
        'intermediator': intermediator_config.get('name'),
        'participant1': participant1_config.get('name'),
        'participant2': participant2_config.get('name')
    })

    # Start dialog in thread
    thread = threading.Thread(
        target=run_dialog_thread,
        args=(intermediator_client, participant1_client, participant2_client,
              context_file, dialog_id, max_turns, session_id, thinking_params,
              prompt_config, intermediator_config, participant1_config, participant2_config,
              enable_tts),
        daemon=True
    )
    thread.start()


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description='Intermediated Dialog (IDi) System')
    parser.add_argument('--port', type=int, default=5006, help='Port to run web server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    # Disable Flask's default access logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print(f"Starting Intermediated Dialog (IDi) server on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
