"""
Text-to-Speech generation using OpenAI API.
"""
import os
import re
import json
from pathlib import Path
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def _load_tts_config() -> dict:
    """Load TTS configuration from server_config.json."""
    try:
        config_path = Path(__file__).parent / 'server_config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            return cfg.get('tts', {})
    except Exception:
        pass
    return {}


def _split_text_at_sentence_boundaries(text: str, max_length: int = 4000) -> list:
    """Split text into chunks of <= max_length characters at sentence boundaries.

    Splits at sentence-ending punctuation (. ! ?) followed by a space or end of string.
    Falls back to splitting at the last space if no sentence boundary is found.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        segment = remaining[:max_length]

        split_pos = -1
        for i in range(len(segment) - 1, -1, -1):
            if segment[i] in '.!?' and (i == len(segment) - 1 or segment[i + 1] == ' '):
                split_pos = i + 1
                break

        if split_pos == -1:
            last_space = segment.rfind(' ')
            if last_space > 0:
                split_pos = last_space
            else:
                split_pos = max_length

        chunks.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return chunks


def generate_tts_audio(text: str, speaker: str, dialog_id: str, topic: str, sequence: int,
                      socketio=None) -> Optional[str]:
    """Generate TTS audio using OpenAI and save to file.

    Args:
        text: The text to convert to speech
        speaker: 'intermediator', 'participant1', or 'participant2'
        dialog_id: The dialog ID
        topic: The debate topic for folder naming
        sequence: Global sequence number for ordering
        socketio: Optional SocketIO instance for emitting progress events

    Returns:
        Path to the saved audio file, or None if TTS fails
    """
    print(f"[TTS Debug] generate_tts_audio called: speaker={speaker}, dialog_id={dialog_id[:8]}, sequence={sequence}")
    try:
        tts_config = _load_tts_config()
        default_voices = {
            'intermediator': 'alloy',
            'participant1': 'echo',
            'participant2': 'fable'
        }
        voice_map = tts_config.get('voices', default_voices)
        voice = voice_map.get(speaker, 'alloy')

        # Create folder name: Debate_{sanitized_topic}_{dialog_id}
        # Sanitize topic for folder name
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic)[:50]  # Remove special chars, limit length
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)  # Replace spaces/dashes with underscore
        # Include dialog_id to make each debate's folder unique
        folder_name = f"Debate_{sanitized_topic}_{dialog_id[:8]}" if sanitized_topic else f"Debate_{dialog_id[:8]}"

        # Create audio directory
        audio_dir = os.path.join('output', 'audio', folder_name)
        os.makedirs(audio_dir, exist_ok=True)

        # Generate filename with sequence number for proper ordering
        filename = f"{sequence:03d}_{speaker}.mp3"
        filepath = os.path.join(audio_dir, filename)

        chunks = _split_text_at_sentence_boundaries(text)
        tts_model = tts_config.get('model', 'tts-1')
        generated_files = []

        for chunk_idx, chunk_text in enumerate(chunks):
            if len(chunks) == 1:
                part_filename = filename
            else:
                part_filename = f"{sequence:03d}_{speaker}_part{chunk_idx + 1:02d}.mp3"

            part_filepath = os.path.join(audio_dir, part_filename)

            if socketio:
                socketio.emit('tts_progress', {
                    'status': 'generating',
                    'speaker': speaker,
                    'sequence': sequence,
                    'filename': part_filename,
                    'part': chunk_idx + 1,
                    'total_parts': len(chunks)
                })

            print(f"[TTS Debug] Calling OpenAI TTS API for {speaker}, voice={voice}, "
                  f"part {chunk_idx + 1}/{len(chunks)}, text_len={len(chunk_text)}")
            response = openai_client.audio.speech.create(
                model=tts_model,
                voice=voice,
                input=chunk_text
            )
            print(f"[TTS Debug] OpenAI TTS API returned, saving to {part_filepath}")

            with open(part_filepath, 'wb') as f:
                f.write(response.content)
            print(f"[TTS Debug] Audio file saved: {part_filepath}")
            generated_files.append(part_filepath)

            if socketio:
                socketio.emit('tts_progress', {
                    'status': 'complete',
                    'speaker': speaker,
                    'sequence': sequence,
                    'filename': part_filename,
                    'filepath': part_filepath,
                    'part': chunk_idx + 1,
                    'total_parts': len(chunks)
                })

        if len(generated_files) > 1:
            print(f"[TTS Debug] Generated {len(generated_files)} audio parts for {speaker}: "
                  f"{[os.path.basename(f) for f in generated_files]}")

        return generated_files[0] if generated_files else None

    except Exception as e:
        # Emit TTS error event
        if socketio:
            socketio.emit('tts_progress', {
                'status': 'error',
                'speaker': speaker,
                'sequence': sequence,
                'error': str(e)
            })
        print(f"[TTS Debug] ERROR for {speaker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_participant_summaries(dialog_data: dict, participant1_client, participant2_client,
                                   socketio=None) -> dict:
    """Generate argument summaries for each participant after dialog completes.

    Args:
        dialog_data: Complete dialog data dictionary
        participant1_client: Participant 1's client instance
        participant2_client: Participant 2's client instance
        socketio: Optional SocketIO instance for emitting events

    Returns:
        Dictionary with summary information
    """
    from utils import generate_argument_diagram

    try:
        # Extract all participant turns
        p1_turns = [msg['message'] for msg in dialog_data.get('conversation_history', [])
                   if msg.get('speaker') == 'participant1']
        p2_turns = [msg['message'] for msg in dialog_data.get('conversation_history', [])
                   if msg.get('speaker') == 'participant2']

        # Combine turns into full argument text
        p1_full_text = "\n\n".join(p1_turns)
        p2_full_text = "\n\n".join(p2_turns)

        # Prepare summary prompt
        summary_prompt = """Please analyze all your arguments from the debate and provide a structured summary including:

1. Your main thesis/position
2. Key arguments you made (as bullet points)
3. Supporting evidence or reasoning you provided
4. Counter-arguments you addressed

Be concise but comprehensive. Format your response in plain text with clear sections."""

        summary_model = "gpt-4o-mini"

        if socketio:
            socketio.emit('summary_progress', {'status': 'generating', 'participant': 1})

        p1_response = openai_client.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "You are analyzing a debate participant's performance."},
                {"role": "user", "content": f"Here were all the contributions from {participant1_client.name} in the debate:\n\n{p1_full_text}\n\n{summary_prompt}"}
            ]
        )
        p1_summary = p1_response.choices[0].message.content

        if socketio:
            socketio.emit('summary_progress', {'status': 'generating', 'participant': 2})

        p2_response = openai_client.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "You are analyzing a debate participant's performance."},
                {"role": "user", "content": f"Here were all the contributions from {participant2_client.name} in the debate:\n\n{p2_full_text}\n\n{summary_prompt}"}
            ]
        )
        p2_summary = p2_response.choices[0].message.content

        # Save summaries to text files
        topic = dialog_data.get('topic', 'Dialog')
        dialog_id = dialog_data.get('dialog_id', '')[:8]
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic)[:50]
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)
        folder_name = f"Debate_{sanitized_topic}_{dialog_id}" if sanitized_topic else f"Debate_{dialog_id}"
        audio_dir = os.path.join('output', 'audio', folder_name)
        os.makedirs(audio_dir, exist_ok=True)

        p1_summary_path = os.path.join(audio_dir, f"summary_{participant1_client.name}.txt")
        p2_summary_path = os.path.join(audio_dir, f"summary_{participant2_client.name}.txt")

        with open(p1_summary_path, 'w', encoding='utf-8') as f:
            f.write(p1_summary)
        with open(p2_summary_path, 'w', encoding='utf-8') as f:
            f.write(p2_summary)

        # Generate argument structure diagrams
        if socketio:
            socketio.emit('summary_progress', {'status': 'diagrams'})

        p1_diagram = generate_argument_diagram(p1_summary, participant1_client.name, audio_dir)
        p2_diagram = generate_argument_diagram(p2_summary, participant2_client.name, audio_dir)

        if socketio:
            socketio.emit('summaries_generated', {
                'participant1': {
                    'summary': p1_summary,
                    'summary_file': p1_summary_path,
                    'diagram_file': p1_diagram
                },
                'participant2': {
                    'summary': p2_summary,
                    'summary_file': p2_summary_path,
                    'diagram_file': p2_diagram
                }
            })

        return {
            'participant1': {
                'summary': p1_summary,
                'summary_file': p1_summary_path,
                'diagram_file': p1_diagram
            },
            'participant2': {
                'summary': p2_summary,
                'summary_file': p2_summary_path,
                'diagram_file': p2_diagram
            }
        }

    except Exception as e:
        print(f"Error generating participant summaries: {e}")
        if socketio:
            socketio.emit('summary_progress', {'status': 'error', 'error': str(e)})
        return {}
