"""
Utility functions for dialog management.
"""
import re
import os
import json
import requests
from datetime import datetime
from typing import Dict, Optional


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
                         server_config: Dict, topic: str) -> tuple:
    """Save dialog data to JSON and PDF files.

    Returns:
        Tuple of (json_path, pdf_path, base_filename)
    """
    from pdf_generator import generate_pdf_from_dialog

    # Generate filename
    base_filename = generate_filename_from_topic(topic)

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Save JSON
    json_path = os.path.join('output', f'{base_filename}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dialog_data': dialog_data,
            'prompt_config': prompt_config,
            'server_config': server_config,
            'topic': topic,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    # Generate PDF
    pdf_path = generate_pdf_from_dialog(dialog_data, prompt_config, server_config, base_filename)

    return json_path, pdf_path, base_filename


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
