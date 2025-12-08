"""
Flask HTTP routes for Intermediator Dialog System.

Routes handle:
- Main page serving
- Debate library management
- PDF generation
- Server status checking
- File uploads
- Cache management
"""

import os
import json
import re
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from flask import render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
import shutil

from clients.ollama_client import OllamaClient
from clients import get_available_providers, ANTHROPIC_AVAILABLE, OPENAI_AVAILABLE, GEMINI_AVAILABLE
from clients.anthropic_client import AnthropicClient
from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient
from models import IntermediatorDialog
from pdf_generator import generate_pdf_from_dialog
from utils import debug_log, generate_filename_from_topic


def register_routes(app, socketio, state):
    """
    Register all Flask routes with the application.

    Args:
        app: Flask application instance
        socketio: SocketIO instance
        state: Dictionary containing global state:
            - complete_dialog_data: Dialog data for PDF generation
            - gpu_monitoring_data: GPU monitoring data per dialog
            - client_instances: Cached OllamaClient instances
            - uploaded_files: Uploaded file paths by session ID
            - dialog_instances: Active dialog instances
    """

    @app.route('/')
    def index():
        """Serve the main page."""
        return render_template('intermediator_dialog.html')

    @app.route('/live')
    def live_viewer():
        """Serve the live debate viewer page."""
        return render_template('live.html')

    @app.route('/api/providers', methods=['GET'])
    def get_providers():
        """Get available AI providers and their models."""
        providers = {
            'ollama': {
                'available': True,
                'name': 'Ollama (Local)',
                'requires_api_key': False,
                'models': []  # Populated dynamically from server
            },
            'anthropic': {
                'available': ANTHROPIC_AVAILABLE,
                'name': 'Anthropic (Claude)',
                'requires_api_key': True,
                'api_key_env': 'ANTHROPIC_API_KEY',
                'models': list(AnthropicClient.MODELS.keys()) if ANTHROPIC_AVAILABLE else []
            },
            'openai': {
                'available': OPENAI_AVAILABLE,
                'name': 'OpenAI',
                'requires_api_key': True,
                'api_key_env': 'OPENAI_API_KEY',
                'models': list(OpenAIClient.MODELS.keys()) if OPENAI_AVAILABLE else []
            },
            'gemini': {
                'available': GEMINI_AVAILABLE,
                'name': 'Google Gemini',
                'requires_api_key': True,
                'api_key_env': 'GOOGLE_API_KEY',
                'models': list(GeminiClient.MODELS.keys()) if GEMINI_AVAILABLE else []
            }
        }

        # Check which API keys are configured
        import os
        api_keys_status = {
            'ANTHROPIC_API_KEY': bool(os.environ.get('ANTHROPIC_API_KEY')),
            'OPENAI_API_KEY': bool(os.environ.get('OPENAI_API_KEY')),
            'GOOGLE_API_KEY': bool(os.environ.get('GOOGLE_API_KEY')),
        }

        return jsonify({
            'providers': providers,
            'api_keys_configured': api_keys_status
        })

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

    @app.route('/default_prompts', methods=['GET'])
    def get_default_prompts():
        """Retrieve the current default prompts."""
        try:
            prompts_path = Path(__file__).parent / 'default_prompts.json'
            if not prompts_path.exists():
                return jsonify({'error': 'Default prompts file not found'}), 404

            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)

            return jsonify(prompts_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/update_default_prompts', methods=['POST'])
    def update_default_prompts():
        """Update default prompts and archive the previous version."""
        try:
            new_prompts = request.json.get('prompts')
            if not new_prompts:
                return jsonify({'error': 'No prompts provided'}), 400

            # Validate required fields
            required_fields = ['intermediator_pre_prompt', 'participant_pre_prompt', 'participant_post_prompt']
            for field in required_fields:
                if field not in new_prompts:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            prompts_path = Path(__file__).parent / 'default_prompts.json'
            archive_dir = Path(__file__).parent / 'archive' / 'default_prompts'
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Archive current version if it exists
            if prompts_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_path = archive_dir / f'default_prompts_{timestamp}.json'
                shutil.copy2(prompts_path, archive_path)
                debug_log('info', f"Archived default prompts to {archive_path}")

            # Update the prompts file
            updated_data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'prompts': new_prompts
            }

            with open(prompts_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)

            debug_log('info', 'Default prompts updated successfully')
            return jsonify({
                'success': True,
                'message': 'Default prompts updated successfully',
                'last_updated': updated_data['last_updated']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/default_prompts_archive', methods=['GET'])
    def get_default_prompts_archive():
        """List all archived versions of default prompts."""
        try:
            archive_dir = Path(__file__).parent / 'archive' / 'default_prompts'
            if not archive_dir.exists():
                return jsonify({'archives': []})

            archives = []
            for archive_file in sorted(archive_dir.glob('default_prompts_*.json'), reverse=True):
                with open(archive_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                archives.append({
                    'filename': archive_file.name,
                    'timestamp': data.get('last_updated', ''),
                    'version': data.get('version', ''),
                })

            return jsonify({'archives': archives})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/download_pdf/<path:filename>', methods=['GET'])
    def download_pdf(filename):
        """Serve a PDF file from the output directory."""
        try:
            output_dir = Path(__file__).parent / 'output'
            pdf_path = output_dir / filename

            # Security check: ensure the file is within the output directory
            if not pdf_path.is_relative_to(output_dir):
                return jsonify({'error': 'Invalid file path'}), 403

            if not pdf_path.exists():
                return jsonify({'error': 'PDF file not found'}), 404

            return send_file(
                str(pdf_path),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/generate_pdf/<dialog_id>', methods=['GET'])
    def generate_pdf(dialog_id):
        """Generate and serve PDF for a completed dialog."""
        if dialog_id not in state['complete_dialog_data']:
            return jsonify({'error': 'Dialog not found or not completed'}), 404

        data = state['complete_dialog_data'][dialog_id]

        # Build server_config dict for pdf_generator
        server_config = {
            'intermediator': data['intermediator_config'],
            'participant1': data['participant1_config'],
            'participant2': data['participant2_config']
        }

        # Generate filename from topic
        topic = data['prompt_config'].get('intermediator_topic_prompt', 'Dialog')
        base_filename = generate_filename_from_topic(topic)

        # Generate PDF
        pdf_path = generate_pdf_from_dialog(
            data['dialog_data'],
            data['prompt_config'],
            server_config,
            base_filename,
            data
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
            for key, client in state['client_instances'].items():
                client.reset_conversation()
            debug_log('info', f"Reset conversation cache for all clients ({len(state['client_instances'])} clients)")
            return jsonify({'message': f'Reset cache for all {len(state["client_instances"])} clients', 'reset_count': len(state['client_instances'])})
        elif client_keys:
            # Reset specific clients
            reset_count = 0
            for key in client_keys:
                if key in state['client_instances']:
                    state['client_instances'][key].reset_conversation()
                    reset_count += 1
            debug_log('info', f"Reset conversation cache for {reset_count} specified clients")
            return jsonify({'message': f'Reset cache for {reset_count} clients', 'reset_count': reset_count})
        else:
            return jsonify({'error': 'No clients specified'}), 400

    @app.route('/gpu_monitoring/<dialog_id>', methods=['GET'])
    def get_gpu_monitoring_data(dialog_id):
        """Retrieve GPU monitoring data for a dialog."""
        import time

        if dialog_id not in state['gpu_monitoring_data']:
            return jsonify({'error': 'GPU monitoring data not found for this dialog'}), 404

        data = state['gpu_monitoring_data'][dialog_id]
        return jsonify({
            'dialog_id': dialog_id,
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time'),
            'duration': data.get('end_time', time.time()) - data.get('start_time', 0),
            'sample_count': len(data.get('samples', [])),
            'samples': data.get('samples', []),
            'server_configs': data.get('server_configs', {})
        })

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
            state['uploaded_files'][session_id] = temp_path
            debug_log('info', f"File uploaded: {file.filename} -> {temp_path} (session_id: {session_id})")
            return jsonify({'session_id': session_id, 'filename': file.filename})
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    @socketio.on('reset_dialog_cache')
    def handle_reset_cache(data):
        """Reset conversation cache for dialog participants."""
        from flask_socketio import emit

        reset_all = data.get('reset_all', False)
        intermediator_key = data.get('intermediator_key')
        participant1_key = data.get('participant1_key')
        participant2_key = data.get('participant2_key')

        reset_count = 0

        if reset_all:
            for key, client in state['client_instances'].items():
                client.reset_conversation()
                reset_count += 1
            emit('cache_reset', {'message': f'Reset cache for all {reset_count} clients', 'reset_count': reset_count})
        else:
            keys_to_reset = []
            if intermediator_key and intermediator_key in state['client_instances']:
                keys_to_reset.append(intermediator_key)
            if participant1_key and participant1_key in state['client_instances']:
                keys_to_reset.append(participant1_key)
            if participant2_key and participant2_key in state['client_instances']:
                keys_to_reset.append(participant2_key)

            for key in keys_to_reset:
                state['client_instances'][key].reset_conversation()
                reset_count += 1

            emit('cache_reset', {'message': f'Reset cache for {reset_count} clients', 'reset_count': reset_count})

    # =========================================================================
    # CONTEXT VIEWER ROUTES
    # =========================================================================

    @app.route('/context')
    def context_viewer_index():
        """Serve the context viewer page."""
        return render_template('context_viewer.html')

    @app.route('/context/single')
    def context_viewer_single():
        """Serve a single-client context viewer page."""
        client_key = request.args.get('client', '')
        role = request.args.get('role', 'unknown')
        return render_template('context_viewer_single.html', client_key=client_key, role=role)

    @app.route('/api/context/clients', methods=['GET'])
    def list_all_clients():
        """List all cached client instances with basic info."""
        clients = {}
        for key, client in state['client_instances'].items():
            clients[key] = {
                'name': client.name,
                'model': client.model,
                'host': client.host,
                'role': getattr(client, 'role', None),
                'message_count': len(client.messages),
                'total_tokens': client.total_tokens,
                'thinking_enabled': client.thinking,
                'reasoning_effort': getattr(client, 'reasoning_effort', None)
            }
        return jsonify({'clients': clients})

    @app.route('/api/context/client/<path:client_key>', methods=['GET'])
    def get_client_context(client_key):
        """Get full context for a specific client by cache key."""
        if client_key not in state['client_instances']:
            return jsonify({'error': f'Client not found: {client_key}'}), 404

        client = state['client_instances'][client_key]
        return jsonify(client.get_full_context())

    @app.route('/api/context/dialog/<dialog_id>', methods=['GET'])
    def get_dialog_context(dialog_id):
        """Get context for all participants in a dialog."""
        # Check active dialogs first
        if dialog_id in state['dialog_instances']:
            dialog = state['dialog_instances'][dialog_id]
            return jsonify({
                'dialog_id': dialog_id,
                'status': 'active',
                'topic': dialog.topic,
                'turn_counter': dialog.turn_counter,
                'intermediator': dialog.intermediator.get_full_context(),
                'participant1': dialog.participant1.get_full_context(),
                'participant2': dialog.participant2.get_full_context()
            })

        # Check completed dialogs
        if dialog_id in state['complete_dialog_data']:
            data = state['complete_dialog_data'][dialog_id]
            return jsonify({
                'dialog_id': dialog_id,
                'status': 'completed',
                'dialog_data': data.get('dialog_data'),
                'prompt_config': data.get('prompt_config'),
                'intermediator_config': data.get('intermediator_config'),
                'participant1_config': data.get('participant1_config'),
                'participant2_config': data.get('participant2_config')
            })

        return jsonify({'error': 'Dialog not found'}), 404

    @app.route('/api/context/dialog/<dialog_id>/<role>', methods=['GET'])
    def get_dialog_role_context(dialog_id, role):
        """Get context for a specific participant in a dialog."""
        valid_roles = ['intermediator', 'participant1', 'participant2']
        if role not in valid_roles:
            return jsonify({'error': f'Invalid role. Must be one of: {valid_roles}'}), 400

        if dialog_id in state['dialog_instances']:
            dialog = state['dialog_instances'][dialog_id]
            client = getattr(dialog, role, None)
            if client:
                return jsonify({
                    'dialog_id': dialog_id,
                    'role': role,
                    'status': 'active',
                    **client.get_full_context()
                })

        return jsonify({'error': 'Dialog or role not found'}), 404

    @app.route('/api/context/active_dialogs', methods=['GET'])
    def list_active_dialogs():
        """List all active dialog instances."""
        dialogs = {}
        for dialog_id, dialog in state['dialog_instances'].items():
            dialogs[dialog_id] = {
                'topic': dialog.topic,
                'turn_counter': dialog.turn_counter,
                'mode': dialog.config.mode.value if hasattr(dialog.config, 'mode') else 'unknown',
                'intermediator': dialog.intermediator.name,
                'participant1': dialog.participant1.name,
                'participant2': dialog.participant2.name
            }
        return jsonify({'active_dialogs': dialogs})
