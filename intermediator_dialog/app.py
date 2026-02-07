#!/usr/bin/env python3
"""
Intermediator Dialog System - Main Entry Point
One AI intermediates a dialog between two other AIs.
"""
import argparse
import json
import os
import webbrowser
import time
from pathlib import Path
from flask import Flask
from flask_socketio import SocketIO
from threading import Timer

import config
from routes import register_routes
from socketio_handlers import register_socketio_handlers
from version import __version_full__, __version__, __build__


def _load_app_config() -> dict:
    """Load server_config.json for app settings."""
    try:
        config_path = Path(__file__).parent / 'server_config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

cors_origins = os.environ.get('CORS_ORIGINS', '*')
socketio = SocketIO(app, cors_allowed_origins=cors_origins, max_http_buffer_size=16 * 1024 * 1024)

# Initialize global state
state = {
    'client_instances': config.client_instances,
    'uploaded_files': config.uploaded_files,
    'dialog_instances': config.dialog_instances,
    'dialog_metadata': config.dialog_metadata,
    'complete_dialog_data': config.complete_dialog_data,
    'gpu_monitoring_data': config.gpu_monitoring_data,
    'gpu_monitoring_threads': config.gpu_monitoring_threads
}

# Register routes and handlers
register_routes(app, socketio, state)
register_socketio_handlers(socketio, state)


def open_browser(host, port):
    """Open browser after a short delay to ensure server is ready."""
    url = f"http://{host}:{port}"
    print(f"\n🚀 Opening browser to {url}")
    webbrowser.open(url)


def main():
    """Main entry point for the application."""
    app_config = _load_app_config()
    default_port = app_config.get('services', {}).get('app_port', 5005)

    parser = argparse.ArgumentParser(description='Run the Intermediator Dialog server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=default_port,
                       help=f'Port to bind to (default: {default_port})')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser automatically')

    args = parser.parse_args()

    print("=" * 60)
    print(f"🤖 Intermediator Dialog System v{__version_full__}")
    print("=" * 60)
    print(f"Version: {__version__} (build {__build__})")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("=" * 60)

    # Open browser after server starts (unless disabled)
    if not args.no_browser:
        # Use localhost for browser if binding to 0.0.0.0
        browser_host = 'localhost' if args.host == '0.0.0.0' else args.host
        Timer(1.5, lambda: open_browser(browser_host, args.port)).start()

    # Run the server
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        allow_unsafe_werkzeug=True  # Allow for development
    )


if __name__ == '__main__':
    main()
