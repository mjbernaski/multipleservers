"""
Global configuration and state management.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Global state dictionaries
client_instances = {}  # Cache client instances by key: "{type}:{model}:{name}"
uploaded_files = {}  # Temporarily store uploaded files
dialog_instances = {}  # Store active dialog instances
dialog_metadata = {}  # Store dialog metadata
complete_dialog_data = {}  # Store complete dialog data for PDF generation
gpu_monitoring_data = {}  # Store GPU monitoring data per dialog
gpu_monitoring_threads = {}  # Store GPU monitoring thread control

def reset_client_cache():
    """Reset all cached client instances."""
    global client_instances
    client_instances = {}

def get_client_cache_key(host: str, model: str, name: str) -> str:
    """Generate cache key for client instances."""
    return f"{host}:{model}:{name}"
