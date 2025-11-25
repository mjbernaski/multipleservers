#!/usr/bin/env python3
"""
Refactored Intermediator Dialog System
Clean implementation with phase-aware prompts and consolidated logic.
"""

import os
import re
import time
import uuid
import threading
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, field

from prompt_templates import PromptTemplates, DialogMode, DialogPhase


@dataclass
class DialogConfig:
    """Configuration for a dialog session."""
    mode: DialogMode = DialogMode.EXPLORATION
    max_turns: int = 3
    enable_tts: bool = True
    
    # Participant positions/roles (optional)
    participant1_position: str = None
    participant2_position: str = None
    
    # Custom instructions
    moderator_instructions: str = None
    participant1_instructions: str = None
    participant2_instructions: str = None
    
    # Context settings
    max_context_chars: int = 800  # Increased from 300
    max_message_chars: int = 1000  # Increased from 400


class IntermediatorDialogRefactored:
    """
    Manages a dialog between two AIs with an intermediator.
    
    Refactored version with:
    - Phase-aware prompting
    - Consolidated participant handling
    - Better context management
    - Dialog mode support
    """

    def __init__(
        self,
        intermediator,  # OllamaClient
        participant1,   # OllamaClient
        participant2,   # OllamaClient
        topic: str,
        config: DialogConfig = None,
        dialog_id: str = None,
        tts_callback: Callable = None,  # External TTS handler
    ):
        self.intermediator = intermediator
        self.participant1 = participant1
        self.participant2 = participant2
        self.topic = topic
        self.config = config or DialogConfig()
        self.dialog_id = dialog_id or str(uuid.uuid4())
        self.tts_callback = tts_callback
        
        # Initialize prompt templates for the selected mode
        self.prompts = PromptTemplates(mode=self.config.mode)
        
        # State
        self.conversation_history: List[Dict] = []
        self.turn_counter = 0
        self.audio_sequence = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Callbacks
        self.stream_callback: Optional[Callable] = None
        self.tts_threads: List[threading.Thread] = []
        
        # Participant name mapping for cleaner prompts
        self.names = {
            'intermediator': self.intermediator.name,
            'participant1': self.participant1.name,
            'participant2': self.participant2.name,
        }

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

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    def _build_context_summary(
        self,
        for_participant: str,
        max_chars: int = None
    ) -> str:
        """
        Build context summary prioritizing recent and relevant messages.
        
        Improved version that:
        - Prioritizes last moderator intervention
        - Includes recent exchanges (not just arbitrary truncation)
        - Uses actual participant names
        """
        max_chars = max_chars or self.config.max_context_chars
        
        if not self.conversation_history:
            return ""
        
        # Get the other participant
        other = 'participant2' if for_participant == 'participant1' else 'participant1'
        
        # Build priority-weighted messages
        priority_messages = []
        
        # Always include last moderator message (high priority)
        for msg in reversed(self.conversation_history):
            if msg['speaker'] == 'intermediator' and not msg.get('is_summary'):
                priority_messages.append(('high', msg))
                break
        
        # Include last message from other participant (high priority)
        for msg in reversed(self.conversation_history):
            if msg['speaker'] == other:
                priority_messages.append(('high', msg))
                break
        
        # Add recent context (lower priority, can be condensed)
        recent = [m for m in self.conversation_history[-4:] 
                  if m not in [pm[1] for pm in priority_messages]]
        for msg in recent:
            priority_messages.append(('medium', msg))
        
        # Format messages with actual names
        formatted = []
        total_chars = 0
        
        for priority, msg in priority_messages:
            speaker_name = self.names.get(msg['speaker'], msg['speaker'])
            content = msg['message']
            
            # Truncate individual messages if needed
            if priority == 'medium' and len(content) > 200:
                content = content[:200] + "..."
            elif len(content) > 400:
                content = content[:400] + "..."
            
            line = f"{speaker_name}: {content}"
            
            if total_chars + len(line) <= max_chars:
                formatted.append(line)
                total_chars += len(line)
            elif priority == 'high':
                # Always include high-priority, even if we need to truncate
                remaining = max_chars - total_chars
                if remaining > 100:
                    formatted.append(line[:remaining] + "...")
        
        return "\n\n".join(formatted)

    def _get_last_relevant_message(
        self,
        exclude_speaker: str
    ) -> Tuple[str, str]:
        """Get the last message from anyone except the specified speaker."""
        for msg in reversed(self.conversation_history):
            if msg['speaker'] != exclude_speaker:
                speaker_name = self.names.get(msg['speaker'], msg['speaker'])
                return msg['message'], speaker_name
        return "", ""

    # =========================================================================
    # PARTICIPANT TURN HANDLING (CONSOLIDATED)
    # =========================================================================

    def _handle_participant_turn(
        self,
        participant_num: int,
        turn: int,
        total_turns: int
    ) -> Tuple[str, Dict]:
        """
        Handle a single participant's turn.
        
        Consolidated method replacing duplicate code for participant1 and participant2.
        """
        # Select the right participant
        if participant_num == 1:
            participant = self.participant1
            speaker_key = 'participant1'
            exclude_speakers = ['participant1']
        else:
            participant = self.participant2
            speaker_key = 'participant2'
            exclude_speakers = ['participant2']
        
        participant_name = self.names[speaker_key]
        
        # Get context and last message
        context_summary = self._build_context_summary(for_participant=speaker_key)
        last_message, last_speaker_name = self._get_last_relevant_message(
            exclude_speaker=speaker_key
        )
        
        # Truncate last message if needed
        max_msg = self.config.max_message_chars
        if len(last_message) > max_msg:
            last_message = last_message[:max_msg] + "..."
        
        # Build the prompt using templates
        prompt = self.prompts.get_participant_turn_prompt(
            context_summary=context_summary,
            last_message=last_message,
            last_speaker_name=last_speaker_name,
            turn=turn,
            total_turns=total_turns
        )
        
        # Emit turn start event
        self._emit('participant_turn', {
            'turn': turn,
            'speaker': speaker_key,
            'message': prompt,
            'thinking': participant.thinking,
            'participant_name': participant_name
        })
        
        # Get response
        response, tokens = participant.ask(prompt, round_num=turn)
        
        # Record in history
        self.conversation_history.append({
            'turn': self.turn_counter,
            'speaker': speaker_key,
            'message': response,
            'tokens': tokens,
            'thinking_enabled': participant.thinking
        })
        self.turn_counter += 1
        
        # Update other participants' context
        self._update_contexts(speaker_key, response)
        
        # Handle TTS if enabled
        if self.config.enable_tts and self.tts_callback:
            self._queue_tts(response, speaker_key)
        
        return response, tokens

    def _update_contexts(self, speaker_key: str, message: str):
        """Update all participants' message histories with a new message."""
        speaker_name = self.names[speaker_key]
        content = f"{speaker_name} said: {message}"
        
        # Update intermediator
        self.intermediator.messages.append({
            "role": "user",
            "content": content
        })
        
        # Update the OTHER participant (not the one who just spoke)
        if speaker_key == 'participant1':
            self.participant2.messages.append({
                "role": "user",
                "content": content
            })
        else:
            self.participant1.messages.append({
                "role": "user",
                "content": content
            })

    # =========================================================================
    # MODERATION HANDLING
    # =========================================================================

    def _handle_moderation(
        self,
        turn: int,
        total_turns: int,
        last_speaker: str,
        last_response: str
    ) -> Tuple[str, Dict, bool]:
        """
        Handle moderator intervention.
        
        Returns: (response, tokens, should_continue)
        """
        speaker_name = self.names[last_speaker]
        
        # Get phase-aware moderation prompt
        mod_prompt = self.prompts.get_moderation_prompt(
            speaker_name=speaker_name,
            turn=turn,
            total_turns=total_turns
        )
        
        self._emit('intermediator_turn', {
            'turn': turn,
            'speaker': 'intermediator',
            'message': mod_prompt
        })
        
        mod_response, mod_tokens = self.intermediator.ask(mod_prompt, round_num=turn)
        
        # Record in history
        self.conversation_history.append({
            'turn': self.turn_counter,
            'speaker': 'intermediator',
            'message': mod_response,
            'tokens': mod_tokens,
            'thinking_enabled': self.intermediator.thinking
        })
        self.turn_counter += 1
        
        # Update participants with moderator's comment
        mod_content = f"Moderator ({self.names['intermediator']}) said: {mod_response}"
        self.participant1.messages.append({"role": "user", "content": mod_content})
        self.participant2.messages.append({"role": "user", "content": mod_content})
        
        # Handle TTS
        if self.config.enable_tts and self.tts_callback:
            self._queue_tts(mod_response, 'intermediator')
        
        # Check for early conclusion signal
        should_continue = "CONCLUDE" not in mod_response.upper()
        
        return mod_response, mod_tokens, should_continue

    # =========================================================================
    # TTS HANDLING
    # =========================================================================

    def _queue_tts(self, text: str, speaker: str):
        """Queue TTS generation in background thread."""
        if not self.tts_callback:
            return
        
        thread = threading.Thread(
            target=self.tts_callback,
            args=(text, speaker, self.dialog_id, self.topic, self.audio_sequence),
            daemon=True
        )
        thread.start()
        self.tts_threads.append(thread)
        self.audio_sequence += 1

    def _wait_for_tts(self, timeout: int = 30):
        """Wait for all TTS threads to complete."""
        for thread in self.tts_threads:
            if thread.is_alive():
                thread.join(timeout=timeout)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _initialize_system_prompts(self, context_content: str = None):
        """Set up system prompts for all participants."""
        
        # Moderator system prompt
        moderator_prompt = self.prompts.get_moderator_system_prompt(
            topic=self.topic,
            custom_instructions=self.config.moderator_instructions
        )
        self.intermediator.messages = [{"role": "system", "content": moderator_prompt}]
        
        # Participant 1 system prompt
        p1_role = "questioner" if self.config.mode == DialogMode.INTERVIEW else None
        p1_prompt = self.prompts.get_participant_system_prompt(
            participant_name=self.names['participant1'],
            position=self.config.participant1_position,
            role=p1_role,
            custom_instructions=self.config.participant1_instructions
        )
        self.participant1.messages = [{"role": "system", "content": p1_prompt}]
        
        # Participant 2 system prompt
        p2_role = "subject" if self.config.mode == DialogMode.INTERVIEW else None
        p2_prompt = self.prompts.get_participant_system_prompt(
            participant_name=self.names['participant2'],
            position=self.config.participant2_position,
            role=p2_role,
            custom_instructions=self.config.participant2_instructions
        )
        self.participant2.messages = [{"role": "system", "content": p2_prompt}]
        
        # Add context if provided
        if context_content:
            context_msg = f"Here is context for our discussion:\n\n{context_content}\n\nPlease use this context to inform your responses."
            self.intermediator.messages.append({"role": "user", "content": context_msg})
            self.participant1.messages.append({"role": "user", "content": context_msg})
            self.participant2.messages.append({"role": "user", "content": context_msg})

    def _preload_models(self):
        """Preload models in parallel (non-blocking)."""
        self._emit('status_update', {
            'message': 'Preloading models on all servers...'
        })
        
        for client, name in [
            (self.intermediator, 'Intermediator'),
            (self.participant1, 'Participant 1'),
            (self.participant2, 'Participant 2')
        ]:
            thread = threading.Thread(target=client.preload_model, daemon=True)
            thread.start()

    # =========================================================================
    # MAIN DIALOG LOOP
    # =========================================================================

    def run_dialog(self, context_file: str = None) -> Dict:
        """
        Run a dialog between the two participants, mediated by the intermediator.
        
        Refactored version with:
        - Phase-aware prompting
        - Consolidated participant handling
        - Clean separation of concerns
        """
        self.start_time = time.time()
        
        # Load context if provided
        context_content = None
        if context_file and os.path.exists(context_file):
            with open(context_file, 'r', encoding='utf-8') as f:
                context_content = f.read()
        
        # Initialize
        self._initialize_system_prompts(context_content)
        self._preload_models()
        
        self._emit('dialog_started', {
            'intermediator': self.names['intermediator'],
            'participant1': self.names['participant1'],
            'participant2': self.names['participant2'],
            'mode': self.config.mode.value
        })
        
        # === INTRODUCTION ===
        intro_prompt = self.prompts.get_intro_prompt(
            participant1_name=self.names['participant1'],
            participant2_name=self.names['participant2']
        )
        
        self._emit('intermediator_turn', {
            'turn': 0,
            'speaker': 'intermediator',
            'message': intro_prompt,
            'thinking': False,
            'intermediator': self.names['intermediator']
        })
        
        intro_response, intro_tokens = self.intermediator.ask(intro_prompt, round_num=0)
        
        self.conversation_history.append({
            'turn': self.turn_counter,
            'speaker': 'intermediator',
            'message': intro_response,
            'tokens': intro_tokens,
            'thinking_enabled': self.intermediator.thinking
        })
        self.turn_counter += 1
        
        # Share intro with participants
        intro_content = f"Moderator ({self.names['intermediator']}) said: {intro_response}"
        self.participant1.messages.append({"role": "user", "content": intro_content})
        self.participant2.messages.append({"role": "user", "content": intro_content})
        
        if self.config.enable_tts and self.tts_callback:
            self._queue_tts(intro_response, 'intermediator')
        
        # === MAIN DIALOG LOOP ===
        total_turns = self.config.max_turns * 2
        
        for turn in range(1, total_turns + 1):
            # Determine which participant's turn
            participant_num = 1 if turn % 2 == 1 else 2
            speaker_key = f'participant{participant_num}'
            
            # Participant responds
            response, tokens = self._handle_participant_turn(
                participant_num=participant_num,
                turn=turn,
                total_turns=total_turns
            )
            
            # Moderator intervenes (skip on final turn)
            if turn < total_turns:
                mod_response, mod_tokens, should_continue = self._handle_moderation(
                    turn=turn,
                    total_turns=total_turns,
                    last_speaker=speaker_key,
                    last_response=response
                )
                
                # Check for early conclusion
                if not should_continue:
                    self._emit('early_conclusion', {
                        'turn': turn,
                        'reason': mod_response
                    })
                    break
        
        # === FINAL SUMMARY ===
        summary_prompt = self.prompts.get_summary_prompt(
            participant1_name=self.names['participant1'],
            participant2_name=self.names['participant2']
        )
        
        self._emit('intermediator_turn', {
            'turn': self.turn_counter,
            'speaker': 'intermediator',
            'message': summary_prompt,
            'is_summary': True,
            'thinking': True,
            'intermediator': self.names['intermediator']
        })
        
        summary_response, summary_tokens = self.intermediator.ask(summary_prompt, round_num=self.turn_counter)
        
        self.conversation_history.append({
            'turn': self.turn_counter,
            'speaker': 'intermediator',
            'message': summary_response,
            'tokens': summary_tokens,
            'is_summary': True,
            'thinking_enabled': self.intermediator.thinking
        })
        self.turn_counter += 1
        
        if self.config.enable_tts and self.tts_callback:
            self._queue_tts(summary_response, 'intermediator')
        
        # Share summary with participants
        summary_content = f"Moderator's Final Summary: {summary_response}"
        self.participant1.messages.append({"role": "user", "content": summary_content})
        self.participant2.messages.append({"role": "user", "content": summary_content})
        
        # Wait for TTS to complete
        self._wait_for_tts()
        
        # Finalize
        self.end_time = time.time()
        runtime_seconds = self.end_time - self.start_time
        
        self._emit('dialog_complete', {
            'conversation_history': self.conversation_history,
            'runtime_seconds': runtime_seconds,
            'total_turns': len(self.conversation_history)
        })
        
        return {
            'conversation_history': self.conversation_history,
            'runtime_seconds': runtime_seconds,
            'total_turns': len(self.conversation_history),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'mode': self.config.mode.value
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of how to use the refactored dialog system."""
    
    # Assuming OllamaClient is available
    # from your_module import OllamaClient
    
    # Create clients (placeholder)
    # intermediator = OllamaClient("http://localhost:11434", "llama3", "Moderator")
    # participant1 = OllamaClient("http://localhost:11434", "llama3", "Alice")
    # participant2 = OllamaClient("http://localhost:11434", "llama3", "Bob")
    
    # Configure a debate
    config = DialogConfig(
        mode=DialogMode.DEBATE,
        max_turns=4,
        enable_tts=False,
        participant1_position="AI systems can achieve genuine understanding",
        participant2_position="AI systems can only simulate understanding",
    )
    
    # Create dialog
    # dialog = IntermediatorDialogRefactored(
    #     intermediator=intermediator,
    #     participant1=participant1,
    #     participant2=participant2,
    #     topic="Can AI achieve genuine understanding, or only simulate it?",
    #     config=config
    # )
    
    # Run dialog
    # result = dialog.run_dialog()
    
    print("Example configuration created successfully.")
    print(f"Mode: {config.mode.value}")
    print(f"Max turns: {config.max_turns}")
    print(f"P1 position: {config.participant1_position}")
    print(f"P2 position: {config.participant2_position}")


if __name__ == "__main__":
    example_usage()
