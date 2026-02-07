# Intermediated Dialog (IDi)

A multi-AI dialog system where one AI intermediates a conversation between two other AIs. The intermediator acts as a moderator, guiding the conversation and ensuring both participants have opportunities to contribute.

## Features

- **Three-AI System**: One intermediator (moderator) and two participants
- **Multiple Providers**: Ollama (local), OpenAI, Anthropic (Claude), Google Gemini
- **Dialog Modes**: Debate, Exploration, Interview, Critique
- **Phase-Aware Prompting**: Moderation adapts across early/middle/late phases
- **Draft-Critique-Final Pipeline**: Participants refine responses before submission
- **Web Interface**: Real-time dialog display with streaming responses
- **TTS Support**: Text-to-speech audio generation with sentence-boundary chunking
- **Context Support**: Upload context files to inform the discussion
- **Configurable**: All server addresses and defaults in `server_config.json`
- **Real-time Updates**: See responses stream in as they're generated
- **Pause/Resume/Stop**: Control dialog flow in real-time
- **Cost Tracking**: Running cost display for cloud API dialogs

## Project Structure

```
intermediator_dialog/
├── app.py                              # Flask entry point
├── server_config.json                  # Server addresses, services, defaults
├── intermediator_dialog_refactored.py  # Dialog engine with phase-aware prompts
├── prompt_templates.py                 # Centralized prompt management
├── clients/                            # LLM provider clients
│   ├── base_client.py                  # Abstract base with retry logic
│   ├── ollama_client.py                # Ollama (local)
│   ├── anthropic_client.py             # Anthropic (Claude)
│   ├── openai_client.py                # OpenAI
│   └── gemini_client.py                # Google Gemini
├── templates/                          # HTML templates
│   └── intermediator_dialog.html       # Main web interface
├── output/                             # Saved dialogs (JSON, TXT, PDF, audio)
├── requirements.txt                    # Python dependencies
├── run_intermediator_dialog.sh         # Startup script
└── README.md                           # This file
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
# Using the startup script (recommended)
./run_intermediator_dialog.sh

# Or directly with Python
python app.py --host 0.0.0.0 --port 5005
```

The web interface will be available at `http://localhost:5005`.

### Configuration

Server addresses and defaults are in `server_config.json`:

```json
{
  "servers": {
    "intermediator": { "host": "http://...:11435", "name": "RT5090", "provider": "ollama", "default_model": "gpt-oss:20b" },
    "participant1": { "host": "http://...:11434", "name": "NVIDIA DGX Spark A", "provider": "ollama", "default_model": "gpt-oss:20b" },
    "participant2": { "host": "http://...:11434", "name": "NVIDIA DGX Spark B", "provider": "ollama", "default_model": "gpt-oss:120b" }
  },
  "services": { "diagram_service": "http://...:7777", "gpu_monitor_port": 9999, "app_port": 5005 },
  "defaults": { "max_turns": 3 },
  "tts": { "model": "tts-1", "voices": { "intermediator": "alloy", "participant1": "echo", "participant2": "fable" } }
}
```

### Using the Interface

1. **Configure Prompt**: Enter the Intermediator Topic/Instructions Prompt (required)
2. **Select Dialog Mode**: Debate, Exploration, Interview, or Critique
3. **Configure AIs** (optional): Adjust host, model, and name for each AI
4. **Upload Context** (optional): Upload a file to provide context for the discussion
5. **Set Max Turns**: Configure how many turns the dialog should have
6. **Start Dialog**: Click "Start Dialog" to begin

### How It Works

1. The intermediator introduces the topic and starts the conversation
2. Participants alternate turns using a draft -> critique -> final response pipeline
3. The intermediator moderates with phase-aware prompts (early/middle/late)
4. All messages are passed between participants to maintain context
5. The intermediator provides a final summary at the end

## Requirements

- Flask >= 2.3.0
- Flask-SocketIO >= 5.3.0
- Requests >= 2.31.0
- Python-dotenv >= 1.0.0
- OpenAI >= 1.0.0 (for TTS)
- Anthropic >= 0.21.0 (optional, for Claude)
- Google-generativeai >= 0.5.0 (optional, for Gemini)
- ReportLab >= 4.0.0 (for PDF generation)
