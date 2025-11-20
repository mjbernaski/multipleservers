"""
Text-to-Speech generation using OpenAI API.
"""
import os
import re
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client for TTS
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


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
        if socketio:
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
        if socketio:
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
        if socketio:
            socketio.emit('tts_progress', {
                'status': 'error',
                'speaker': speaker,
                'sequence': sequence,
                'error': str(e)
            })
        print(f"TTS Error for {speaker}: {str(e)}")
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

        # Generate summaries using each participant's own client
        if socketio:
            socketio.emit('summary_progress', {'status': 'generating', 'participant': 1})

        # Reset conversation for summary generation
        original_p1_messages = participant1_client.messages.copy()
        participant1_client.reset_conversation()
        participant1_client.messages = [
            {"role": "system", "content": "You are analyzing your own debate performance."},
            {"role": "user", "content": f"Here were all your contributions to the debate:\n\n{p1_full_text}\n\n{summary_prompt}"}
        ]
        p1_summary, _ = participant1_client.ask("Generate the summary now.", round_num=0)
        participant1_client.messages = original_p1_messages  # Restore

        if socketio:
            socketio.emit('summary_progress', {'status': 'generating', 'participant': 2})

        original_p2_messages = participant2_client.messages.copy()
        participant2_client.reset_conversation()
        participant2_client.messages = [
            {"role": "system", "content": "You are analyzing your own debate performance."},
            {"role": "user", "content": f"Here were all your contributions to the debate:\n\n{p2_full_text}\n\n{summary_prompt}"}
        ]
        p2_summary, _ = participant2_client.ask("Generate the summary now.", round_num=0)
        participant2_client.messages = original_p2_messages  # Restore

        # Save summaries to text files
        topic = dialog_data.get('topic', 'Dialog')
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic)[:50]
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)
        folder_name = f"Debate_{sanitized_topic}" if sanitized_topic else "Debate"
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
