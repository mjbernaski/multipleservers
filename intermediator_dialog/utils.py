"""
Utility functions for dialog management.
"""
import re
import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def load_default_prompts() -> Dict[str, str]:
    """
    Load default prompts from external configuration file.

    Returns:
        Dictionary containing default prompts with keys:
        - intermediator_pre_prompt
        - participant_pre_prompt
        - participant_post_prompt
    """
    default_fallbacks = {
        'intermediator_pre_prompt': """You are the intermediator for a dialog between two AI participants.
Your role is to present the participants with the rules and the topic.
You review each response before passing
it on to the other participant.
If you have to remind the participants of the rules you will.
If a participant misbehaves you will mention that in your final summary.
encourage brevity and clarity.""",
        'participant_pre_prompt': """You are the participant in a debate.
You follow the instructions of the intermediator, review the responses of your adversary, and develop thoughtful on-point responses.""",
        'participant_post_prompt': """You bottom line your response in 1 sentence."""
    }

    try:
        prompts_path = Path(__file__).parent / 'default_prompts.json'
        if prompts_path.exists():
            with open(prompts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'prompts' in data:
                    return data['prompts']
    except Exception as e:
        debug_log('warning', f'Failed to load default prompts from file: {e}, using fallback defaults')

    return default_fallbacks


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
        
        print(f"✓ Dialog saved to {json_path} and {txt_path}")
        return str(json_path), str(txt_path)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in save_dialog_to_files: {str(e)}", flush=True)
        print(f"Traceback: {error_details}", flush=True)
        return None, None


def generate_argument_diagram(summary_text: str, participant_name: str, output_dir: str) -> Optional[str]:
    """Generate argument structure diagram by posting to diagram service.

    Args:
        summary_text: The participant's argument summary
        participant_name: Name of the participant
        output_dir: Directory to save diagram

    Returns:
        Path to saved diagram file, or None if failed
    """
    try:
        diagram_service_url = "http://192.168.6.202:7777"

        response = requests.post(
            diagram_service_url,
            json={'text': summary_text, 'orientation': 'horizontal'},
            timeout=30
        )

        if response.status_code == 200:
            # Save diagram as PNG
            diagram_filename = f"diagram_{participant_name}.png"
            diagram_path = os.path.join(output_dir, diagram_filename)

            with open(diagram_path, 'wb') as f:
                f.write(response.content)

            print(f"✓ Generated argument diagram for {participant_name}")
            return diagram_path
        else:
            print(f"Warning: Diagram service returned status {response.status_code}")
            return None

    except Exception as e:
        print(f"Warning: Failed to generate diagram for {participant_name}: {e}")
        return None


def debug_log(level: str, message: str, server: str = None, data: dict = None, socketio=None):
    """Emit debug log event via WebSocket and print to console."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message,
        'server': server
    }
    if data:
        log_entry['data'] = data

    if socketio:
        socketio.emit('debug_log', log_entry)

    server_prefix = f"[{server}] " if server else ""
    print(f"[{timestamp}] [{level.upper()}] {server_prefix}{message}", flush=True)
