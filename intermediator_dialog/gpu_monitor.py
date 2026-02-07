"""
GPU monitoring for Ollama servers.
"""
import json
import time
import requests
import threading
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def _get_gpu_monitor_port() -> int:
    """Read GPU monitor port from server_config.json."""
    try:
        config_path = Path(__file__).parent / 'server_config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            return cfg.get('services', {}).get('gpu_monitor_port', 9999)
    except Exception:
        pass
    return 9999


def fetch_gpu_status(host: str) -> Optional[Dict]:
    """Fetch GPU status from a server's monitoring endpoint.

    Args:
        host: The host URL (e.g., 'http://192.168.5.40:11434')

    Returns:
        Dictionary with GPU status data or None if unavailable
    """
    try:
        gpu_port = _get_gpu_monitor_port()
        import re
        base_host = re.sub(r':\d+$', '', host)
        gpu_status_url = f"{base_host}:{gpu_port}/gpu-status"

        response = requests.get(gpu_status_url, timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        # Silently fail - GPU monitoring is optional
        return None


def gpu_monitoring_thread(dialog_id: str, server_configs: Dict, gpu_monitoring_data: Dict,
                         gpu_monitoring_threads: Dict, socketio=None, poll_interval: float = 1.0):
    """Background thread to poll GPU status during dialog execution.

    Args:
        dialog_id: The dialog ID to track
        server_configs: Dictionary with 'intermediator', 'participant1', 'participant2' host configs
        gpu_monitoring_data: Shared dictionary to store monitoring data
        gpu_monitoring_threads: Shared dictionary for thread control
        socketio: SocketIO instance for emitting updates
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
        if socketio:
            socketio.emit('gpu_status_update', {
                'dialog_id': dialog_id,
                'sample': sample
            })

        # Sleep until next poll
        time.sleep(poll_interval)

    # Mark end time when monitoring stops
    gpu_monitoring_data[dialog_id]['end_time'] = time.time()


def start_gpu_monitoring(dialog_id: str, intermediator_config: Dict,
                        participant1_config: Dict, participant2_config: Dict,
                        gpu_monitoring_data: Dict, gpu_monitoring_threads: Dict,
                        socketio=None):
    """Start GPU monitoring for a dialog.

    Args:
        dialog_id: The dialog ID
        intermediator_config: Intermediator server configuration
        participant1_config: Participant 1 server configuration
        participant2_config: Participant 2 server configuration
        gpu_monitoring_data: Shared dictionary to store monitoring data
        gpu_monitoring_threads: Shared dictionary for thread control
        socketio: SocketIO instance
    """
    from utils import debug_log

    server_configs = {
        'intermediator': intermediator_config,
        'participant1': participant1_config,
        'participant2': participant2_config
    }

    thread = threading.Thread(
        target=gpu_monitoring_thread,
        args=(dialog_id, server_configs, gpu_monitoring_data, gpu_monitoring_threads, socketio),
        daemon=True
    )
    thread.start()
    debug_log('info', f"Started GPU monitoring for dialog {dialog_id}", socketio=socketio)


def stop_gpu_monitoring(dialog_id: str, gpu_monitoring_threads: Dict, socketio=None):
    """Stop GPU monitoring for a dialog.

    Args:
        dialog_id: The dialog ID
        gpu_monitoring_threads: Shared dictionary for thread control
        socketio: SocketIO instance
    """
    from utils import debug_log

    if dialog_id in gpu_monitoring_threads:
        gpu_monitoring_threads[dialog_id]['running'] = False
        debug_log('info', f"Stopped GPU monitoring for dialog {dialog_id}", socketio=socketio)
