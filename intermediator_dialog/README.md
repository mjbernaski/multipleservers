# Intermediated Dialog (IDi)

A new project that uses one AI to intermediate a dialog between two other AIs. The intermediator acts as a moderator, guiding the conversation and ensuring both participants have opportunities to contribute.

## Features

- **Three-AI System**: One intermediator (moderator) and two participants
- **Default Configuration**: RT5090 is configured as the default intermediator
- **Web Interface**: Real-time dialog display with streaming responses
- **Context Support**: Upload context files to inform the discussion
- **Configurable**: Customize all three AI configurations (host, model, name)
- **Real-time Updates**: See responses stream in as they're generated

## Project Structure

```
intermediator_dialog/
├── intermediator_dialog.py    # Main Flask application
├── templates/
│   └── intermediator_dialog.html  # Web interface
├── output/                     # Output directory for saved dialogs
├── requirements.txt           # Python dependencies
├── run_intermediator_dialog.sh # Startup script
└── README.md                  # This file
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install dependencies (can use the parent project's venv or create a new one):

```bash
# If using parent project's venv
source ../venv/bin/activate
pip install -r requirements.txt

# Or create a new venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
# Using the startup script
./run_intermediator_dialog.sh

# Or directly with Python
python intermediator_dialog.py --host 0.0.0.0 --port 5006
```

The web interface will be available at `http://localhost:5006` (or the configured host/port).

### Default Server Configuration

- **Intermediator**: RT5090 (http://localhost:11434, model: gpt-oss:20b)
- **Participant 1**: NVIDIA DGX Spark 1 (http://localhost:11434, model: gpt-oss:20b)
- **Participant 2**: NVIDIA DGX Spark 2 (http://localhost:11434, model: gpt-oss:120b)

### Using the Interface

1. **Configure Prompt**: Enter the Intermediator Topic/Instructions Prompt (required) - this frames the topic and instructions
2. **Configure AIs** (optional): Adjust host, model, and name for each AI
3. **Upload Context** (optional): Upload a file to provide context for the discussion
4. **Set Max Turns**: Configure how many turns the dialog should have (default: 3)
5. **Start Dialog**: Click "Start Dialog" to begin

### How It Works

1. The intermediator introduces the topic (from its prompt) and starts the conversation
2. Participants alternate turns, responding to each other and the intermediator
3. The intermediator moderates after each participant response, keeping the dialog on track
4. All messages are passed between participants to maintain context
5. The conversation continues for the specified number of turns
6. The intermediator provides a final summary at the end

## Configuration

All three AI configurations can be customized in the web interface:
- **Host**: Ollama server URL (e.g., `http://192.168.6.40:11434`)
- **Model**: Model name on that server (e.g., `gpt-oss:20b`)
- **Name**: Display name for the AI

## Differences from Five Whys

This project is a fork of the Five Whys project but with a different purpose:

- **Five Whys**: Parallel analysis where multiple AIs independently answer the same question
- **Intermediated Dialog (IDi)**: Collaborative dialog where AIs interact with each other through a moderator

## Requirements

- Flask >= 2.3.0
- Flask-SocketIO >= 5.3.0
- Requests >= 2.31.0
- Python-dotenv >= 1.0.0

## Notes

- The intermediator moderates after each participant response to keep the dialog on track
- Participants see each other's responses in their conversation cache
- Each Spark server maintains its conversation cache until explicitly reset
- All responses stream in real-time via WebSocket
- Token usage is tracked and displayed for each response
- A status bar at the bottom shows who currently has the ball (whose turn it is)
