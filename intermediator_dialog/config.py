"""
Global configuration and state management.
"""
import os
import threading
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Thread lock for shared state
_state_lock = threading.Lock()

# Global state dictionaries
client_instances = {}
uploaded_files = {}
dialog_instances = {}
dialog_metadata = {}
complete_dialog_data = {}
gpu_monitoring_data = {}
gpu_monitoring_threads = {}


def get_state_lock() -> threading.Lock:
    """Return the global state lock for thread-safe access."""
    return _state_lock


def reset_client_cache():
    """Reset all cached client instances."""
    global client_instances
    with _state_lock:
        client_instances = {}


def get_client_cache_key(host: str, model: str, name: str) -> str:
    """Generate cache key for client instances."""
    return f"{host}:{model}:{name}"
