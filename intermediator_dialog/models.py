"""
Core dialog management classes.
"""
import os
import time
import threading
import uuid
from typing import Dict, List, Callable, Optional
from clients.base_client import BaseClient
from utils import load_default_prompts


class IntermediatorDialog:
    """Manages a dialog between two AIs with an intermediator."""

    def __init__(self, intermediator: BaseClient, participant1: BaseClient, participant2: BaseClient,
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
        # Import here to avoid circular dependency
        from tts import generate_tts_audio

        self.start_time = time.time()

        # Load context if provided
        context_content = None
        if context_file and os.path.exists(context_file):
            with open(context_file, 'r', encoding='utf-8') as f:
                context_content = f.read()

        # Initialize system prompts
        # Load default prompts from external configuration
        default_prompts = load_default_prompts()

        # Build intermediator prompt from pre-prompt + topic prompt
        no_markdown_instruction = "\n\nIMPORTANT: Use plain text only. Do NOT use markdown formatting (no **bold**, no *italic*, no # headers, no lists with markdown syntax, etc.). Write in natural, plain text."

        if self.intermediator_pre_prompt and self.intermediator_topic_prompt:
            intermediator_system_prompt = f"{self.intermediator_pre_prompt}\n\n{self.intermediator_topic_prompt}{no_markdown_instruction}"
        elif self.intermediator_topic_prompt:
            # If only topic prompt provided, use default pre-prompt
            default_pre = default_prompts.get('intermediator_pre_prompt', '')
            intermediator_system_prompt = f"{default_pre}\n\n{self.intermediator_topic_prompt}{no_markdown_instruction}"
        elif self.intermediator_pre_prompt:
            # If only pre-prompt provided, use it alone
            intermediator_system_prompt = f"{self.intermediator_pre_prompt}{no_markdown_instruction}"
        else:
            # Default prompt if nothing provided
            intermediator_system_prompt = default_prompts.get('intermediator_pre_prompt', '') + no_markdown_instruction

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
                # Default prompt if none provided - use pre and post defaults
                default_parts = []
                if default_prompts.get('participant_pre_prompt'):
                    default_parts.append(default_prompts['participant_pre_prompt'])
                if default_prompts.get('participant_post_prompt'):
                    default_parts.append(default_prompts['participant_post_prompt'])

                if default_parts:
                    return "\n\n".join(default_parts) + no_markdown_instruction
                else:
                    # Ultimate fallback if default_prompts.json can't be loaded
                    return "You are participating in a moderated dialog." + no_markdown_instruction

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
            # Check if client has preload_model method (Ollama-specific)
            if hasattr(client, 'preload_model'):
                thread = threading.Thread(target=client.preload_model, daemon=True)
                thread.start()
                preload_threads.append(thread)

        # NOTE: We do NOT wait for preloads to complete, so the Intermediator can start immediately

        # Start with intermediator introducing the topic/starting the conversation
        intro_prompt = """Please introduce the topic to the two participants and start the conversation. Address both participants and set the stage for a productive discussion."""

        self._emit('intermediator_turn', {
            'turn': 0,
            'speaker': 'intermediator',
            'message': intro_prompt,
            'thinking': True,
            'intermediator': self.intermediator.name
        })

        intro_response, intro_tokens = self.intermediator.ask(intro_prompt, round_num=0)

        self.conversation_history.append({
            'turn': 0,
            'speaker': 'intermediator',
            'message': intro_response,
            'tokens': intro_tokens
        })

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
                    'turn': turn,
                    'speaker': 'participant1',
                    'message': p1_response,
                    'tokens': p1_tokens
                })

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

                # Intermediator moderates after each participant response
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
                        'turn': turn,
                        'speaker': 'intermediator',
                        'message': mod_response,
                        'tokens': mod_tokens
                    })

                    # Generate TTS for intermediator moderation
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
                    'turn': turn,
                    'speaker': 'participant2',
                    'message': p2_response,
                    'tokens': p2_tokens
                })

                # Generate TTS for participant 2
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

                # Intermediator moderates after each participant response
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
                        'turn': turn,
                        'speaker': 'intermediator',
                        'message': mod_response,
                        'tokens': mod_tokens
                    })

                    # Generate TTS for intermediator moderation
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
        summary_turn = len(self.conversation_history)

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
            'turn': summary_turn,
            'speaker': 'intermediator',
            'message': summary_response,
            'tokens': summary_tokens,
            'is_summary': True
        })

        # Generate TTS for final summary
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
