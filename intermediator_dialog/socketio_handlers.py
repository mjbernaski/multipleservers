"""
SocketIO event handlers for Intermediated Dialog system.
"""

import os
import uuid
import threading
from typing import Dict
from datetime import datetime
from flask_socketio import emit
from clients.ollama_client import OllamaClient
from models import IntermediatorDialog
from intermediator_dialog_refactored import IntermediatorDialogRefactored, DialogConfig
from prompt_templates import DialogMode
from gpu_monitor import start_gpu_monitoring, stop_gpu_monitoring
from tts import generate_tts_audio, generate_participant_summaries
from utils import debug_log, save_dialog_to_files, generate_filename_from_topic
from pdf_generator import generate_pdf_from_dialog


def register_socketio_handlers(socketio, state):
    """
    Register all SocketIO event handlers.

    Args:
        socketio: Flask-SocketIO instance
        state: Dictionary containing:
            - client_instances: Dict of cached OllamaClient instances
            - dialog_instances: Dict of active IntermediatorDialog instances
            - dialog_metadata: Dict of dialog metadata
            - complete_dialog_data: Dict of complete dialog data for PDF generation
            - gpu_monitoring_data: Dict of GPU monitoring data
            - gpu_monitoring_threads: Dict of GPU monitoring thread controls
            - uploaded_files: Dict of uploaded file paths by session_id
    """

    client_instances = state['client_instances']
    dialog_instances = state['dialog_instances']
    dialog_metadata = state['dialog_metadata']
    complete_dialog_data = state['complete_dialog_data']
    gpu_monitoring_data = state['gpu_monitoring_data']
    gpu_monitoring_threads = state['gpu_monitoring_threads']
    uploaded_files = state['uploaded_files']
    file_usage_count = state.get('file_usage_count', {})

    def run_dialog_thread(intermediator_client, participant1_client, participant2_client,
                          context_file: str, dialog_id: str, max_turns: int = 3,
                          session_id: str = None, thinking_params: Dict = None,
                          prompt_config: Dict = None, intermediator_config: Dict = None,
                          participant1_config: Dict = None, participant2_config: Dict = None,
                          enable_tts: bool = True):
        """Run dialog in a separate thread."""
        try:
            if thinking_params:
                if 'thinking' in thinking_params:
                    intermediator_client.thinking = thinking_params['thinking']
                    participant1_client.thinking = thinking_params['thinking']
                    participant2_client.thinking = thinking_params['thinking']
                if 'temperature' in thinking_params:
                    intermediator_client.temperature = thinking_params['temperature']
                    participant1_client.temperature = thinking_params['temperature']
                    participant2_client.temperature = thinking_params['temperature']

            prompt_config = prompt_config or {}
            
            # Determine dialog mode from prompt_config
            mode_str = prompt_config.get('dialog_mode', 'exploration')
            mode_map = {
                'debate': DialogMode.DEBATE,
                'exploration': DialogMode.EXPLORATION,
                'interview': DialogMode.INTERVIEW,
                'critique': DialogMode.CRITIQUE
            }
            dialog_mode = mode_map.get(mode_str.lower(), DialogMode.EXPLORATION)
            
            # Create DialogConfig for refactored dialog
            config = DialogConfig(
                mode=dialog_mode,
                max_turns=max_turns,
                enable_tts=enable_tts,
                participant1_position=prompt_config.get('participant1_mid_prompt'),
                participant2_position=prompt_config.get('participant2_mid_prompt'),
                moderator_instructions=prompt_config.get('intermediator_pre_prompt'),
                participant1_instructions=prompt_config.get('participant_pre_prompt'),
                participant2_instructions=prompt_config.get('participant_pre_prompt'),
            )
            
            # Use refactored dialog with phase-aware prompts
            dialog = IntermediatorDialogRefactored(
                intermediator=intermediator_client,
                participant1=participant1_client,
                participant2=participant2_client,
                topic=prompt_config.get('intermediator_topic_prompt', ''),
                config=config,
                dialog_id=dialog_id,
                tts_callback=generate_tts_audio if enable_tts else None
            )
            dialog_instances[dialog_id] = dialog

            dialog.set_stream_callback(lambda data: socketio.emit('dialog_update', data))

            start_gpu_monitoring(
                dialog_id,
                intermediator_config, participant1_config, participant2_config,
                gpu_monitoring_data, gpu_monitoring_threads, socketio
            )

            dialog_result = dialog.run_dialog(context_file=context_file)

            stop_gpu_monitoring(dialog_id, gpu_monitoring_threads, socketio)

            if dialog_result and intermediator_config and participant1_config and participant2_config:
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

                gpu_data = gpu_monitoring_data.get(dialog_id, {})

                complete_dialog_data[dialog_id] = {
                    'dialog_data': dialog_result,
                    'prompt_config': prompt_config,
                    'intermediator_config': intermediator_config,
                    'participant1_config': participant1_config,
                    'participant2_config': participant2_config,
                    'gpu_data': gpu_data,
                    'dialog_id': dialog_id
                }

                # Generate PDF automatically
                server_config = {
                    'intermediator': intermediator_config,
                    'participant1': participant1_config,
                    'participant2': participant2_config
                }

                topic = prompt_config.get('intermediator_topic_prompt', 'Dialog')
                base_filename = generate_filename_from_topic(topic)

                pdf_path = generate_pdf_from_dialog(
                    dialog_result,
                    prompt_config,
                    server_config,
                    base_filename,
                    complete_dialog_data[dialog_id]
                )

                if pdf_path:
                    socketio.emit('pdf_generated', {
                        'dialog_id': dialog_id,
                        'pdf_path': pdf_path,
                        'filename': os.path.basename(pdf_path)
                    })
                    debug_log('info', f"PDF generated successfully: {pdf_path}", socketio=socketio)
                else:
                    debug_log('error', f"Failed to generate PDF for dialog {dialog_id}", socketio=socketio)

                topic = prompt_config.get('intermediator_topic_prompt', 'Debate')
                summary_thread = threading.Thread(
                    target=generate_participant_summaries,
                    args=(dialog_result, participant1_client, participant2_client),
                    kwargs={'socketio': socketio},
                    daemon=True
                )
                summary_thread.start()

        except Exception as e:
            error_msg = str(e)
            socketio.emit('error', {
                'error': error_msg
            })
            socketio.emit('summaries_generated', {
                'dialog_id': dialog_id,
                'summary_path': None,
                'diagram_path': None
            })
            debug_log('error', f"Dialog thread error: {error_msg}", socketio=socketio)
        finally:
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
        enable_tts = data.get('enable_tts', False)

        if not prompt_config.get('intermediator_topic_prompt'):
            emit('error', {'error': 'Intermediator topic/instructions prompt is required'})
            return

        shared_spark_model = data.get('shared_spark_model')
        participant1_override_model = data.get('participant1_override_model')
        participant2_override_model = data.get('participant2_override_model')

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

        if session_id and session_id in uploaded_files:
            context_file = uploaded_files[session_id]
            if context_file and os.path.exists(context_file):
                file_size = os.path.getsize(context_file)
                debug_log('info', f"Using uploaded file: session_id={session_id}, path={context_file}, size={file_size} bytes", socketio=socketio)
            else:
                context_file = None

        dialog_id = str(uuid.uuid4())

        dialog_metadata[dialog_id] = {
            'intermediator_topic_prompt': prompt_config.get('intermediator_topic_prompt', '')[:200] + '...' if prompt_config.get('intermediator_topic_prompt') else None,
            'context_filename': os.path.basename(context_file) if context_file else None,
            'timestamp': datetime.now().isoformat(),
            'intermediator': intermediator_config.get('name'),
            'participant1': participant1_config.get('name'),
            'participant2': participant2_config.get('name'),
            'max_turns': max_turns
        }

        def get_or_create_client(config, role_name):
            """Get existing client instance or create new one to maintain cache."""
            key = f"{config['host']}:{config['model']}:{config.get('name', role_name)}"

            if key in client_instances:
                client = client_instances[key]
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
                debug_log('info', f"Reusing cached client for {role_name}: {key}", server=role_name, socketio=socketio)
                return client
            else:
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
                debug_log('info', f"Created new client for {role_name}: {key}", server=role_name, socketio=socketio)
                return client

        intermediator_client = get_or_create_client(intermediator_config, 'intermediator')
        participant1_client = get_or_create_client(participant1_config, 'participant1')
        participant2_client = get_or_create_client(participant2_config, 'participant2')

        if not intermediator_client.check_server_available():
            emit('error', {'error': f'Intermediator server {intermediator_config["host"]} not available'})
            return
        if not participant1_client.check_server_available():
            emit('error', {'error': f'Participant 1 server {participant1_config["host"]} not available'})
            return
        if not participant2_client.check_server_available():
            emit('error', {'error': f'Participant 2 server {participant2_config["host"]} not available'})
            return

        if session_id and session_id in uploaded_files:
            file_usage_count[session_id] = 1

        emit('dialog_started', {
            'dialog_id': dialog_id,
            'has_context_file': bool(context_file),
            'context_filename': os.path.basename(context_file) if context_file else None,
            'intermediator': intermediator_config.get('name'),
            'participant1': participant1_config.get('name'),
            'participant2': participant2_config.get('name')
        })

        thread = threading.Thread(
            target=run_dialog_thread,
            args=(intermediator_client, participant1_client, participant2_client,
                  context_file, dialog_id, max_turns, session_id, thinking_params,
                  prompt_config, intermediator_config, participant1_config, participant2_config,
                  enable_tts),
            daemon=True
        )
        thread.start()

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
