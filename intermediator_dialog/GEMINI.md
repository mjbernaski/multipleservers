# GEMINI.md

This file provides context and guidance for Gemini agents working on this project.

## Project Overview

**Intermediated Dialog (IDi)** is a Python-based web application that facilitates a three-way conversation between AI models. It features one "Intermediator" (Moderator) and two "Participants". The system orchestrates turn-taking, context management, and real-time streaming of responses to a web interface.

- **Core Function:** Orchestrate dialog between 3 local LLMs (via Ollama).
- **Interface:** Web UI (Flask + SocketIO).
- **Output:** Real-time text streaming, JSON logs, PDF reports, and Text-to-Speech (TTS) audio.

## Architecture

### Key Components

1.  **`intermediator_dialog.py`**: The monolithic entry point containing:
    -   `OllamaClient` class: Wrapper for Ollama API interactions, handling streaming and history.
    -   `IntermediatorDialog` class: Manages the conversation state, turn logic, and file generation.
    -   Flask Application: Routes and SocketIO event handlers.
    -   TTS Logic: Integration with OpenAI's API for audio generation.

2.  **`templates/intermediator_dialog.html`**: The frontend user interface.
    -   Handles real-time updates via WebSockets.
    -   Provides configuration controls for AI endpoints and prompts.

3.  **`output/`**: Storage for generated artifacts.
    -   Subdirectories for `audio/` (MP3 files).
    -   Stores JSON, TXT, and PDF transcripts of dialogs.

### Data Flow

1.  **Setup**: User configures 3 AI endpoints (Intermediator, Participant 1, Participant 2) in the UI.
2.  **Initiation**: User provides a topic. Intermediator generates an intro.
3.  **Loop**:
    -   Participant 1 responds.
    -   Intermediator moderates.
    -   Participant 2 responds.
    -   Intermediator moderates.
4.  **Streaming**: Content is streamed token-by-token to the UI via SocketIO.
5.  **Completion**: After `max_turns`, a summary is generated, and artifacts (PDF, Audio) are saved.

## Development & Usage

### Prerequisites

-   **Python 3.7+**
-   **Ollama**: Running locally or accessible via network (default ports 11434).
-   **OpenAI API Key**: Required for TTS features (set in `.env`).

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Application

Use the provided helper script which handles port conflicts and environment activation:

```bash
./run_intermediator_dialog.sh
```

Or run directly with Python:

```bash
python intermediator_dialog.py --host 0.0.0.0 --port 5006
```

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=sk-... # Required for Text-to-Speech
```

## Codebase Conventions

-   **Style**: Adhere to PEP 8.
-   **State Management**: `client_instances` global dictionary caches `OllamaClient` objects to maintain conversation history across requests.
-   **Async/Concurrency**: Uses `threading` for background generation tasks to avoid blocking the Flask main thread.
-   **File Naming**: Output files use sanitized topic names + timestamps.

## Key Features Implementation

-   **TTS**: `generate_tts_audio` function handles calling OpenAI and saving MP3s.
-   **PDF**: Uses `ReportLab` to generate formatted transcripts.
-   **Thinking Models**: Supports a `thinking` parameter for models that output reasoning traces (e.g., DeepSeek).
