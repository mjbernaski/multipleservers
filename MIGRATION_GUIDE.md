# Migration Guide: Moving to Intel-based Mac

## Overview
This guide helps you transfer the `longchat` project from Apple Silicon (M-series) to an Intel-based Mac.

## Prerequisites on Intel Mac
- Python 3.8 or higher (recommended: Python 3.13 to match current environment)
- Git
- pip

## Step-by-Step Migration

### 1. Transfer Project Files
Choose one of these methods:

#### Option A: Using Git (Recommended)
```bash
# On current Mac: Commit and push any changes
cd /Users/michaelbernaski/Developer/longchat
git add .
git commit -m "Pre-migration commit"
git push origin main

# On Intel Mac: Clone the repository
git clone <your-repo-url> longchat
cd longchat
```

#### Option B: Using rsync or scp
```bash
# From Intel Mac (192.168.6.44), run:
rsync -avz --exclude='.venv' --exclude='venv' --exclude='__pycache__' --exclude='output' --exclude='tts_cache' \
  michaelbernaski@<current-mac-ip>:/Users/michaelbernaski/Developer/longchat/ ~/Developer/longchat/
```

Or from the current Mac, push to Intel Mac:
```bash
# From current Mac, run:
rsync -avz --exclude='.venv' --exclude='venv' --exclude='__pycache__' --exclude='output' --exclude='tts_cache' \
  /Users/michaelbernaski/Developer/longchat/ michaelbernaski@192.168.6.44:~/Developer/longchat/
```

#### Option C: Using external drive
```bash
# Copy the entire project folder, but exclude:
# - .venv/
# - venv/
# - __pycache__/
# - output/
# - tts_cache/
```

### 2. Set Up Python Environment on Intel Mac

```bash
cd ~/Developer/longchat

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy .env file (contains API keys and configuration)
# Make sure to transfer your .env file separately or recreate it
```

The `.env` file should contain your API keys and configuration. Since it's in `.gitignore`, you'll need to:
- Manually copy it from the current Mac, OR
- Recreate it with the necessary credentials

### 4. Architecture Considerations

**Important**: Some Python packages with compiled extensions may have architecture-specific builds. The packages in `requirements.txt` are all pure Python or have universal wheels, so they should work fine on Intel.

If you encounter any issues, you can:
```bash
# Force reinstall all packages
pip install --force-reinstall -r requirements.txt

# Or clear pip cache first
pip cache purge
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Test the connection
python3 test_connection.py

# Run the main application
python3 five_whys_parallel.py
# or
./run_five_whys.sh
```

## What NOT to Transfer

The following directories are excluded from git and should be regenerated on the Intel Mac:
- `.venv/` - Virtual environment (architecture-specific)
- `venv/` - Old virtual environment
- `__pycache__/` - Python cache files
- `output/` - Generated output files
- `tts_cache/` - TTS cache files
- `*.pyc`, `*.pyo` - Compiled Python files

## Troubleshooting

### If packages fail to install:
```bash
# Try installing packages one by one
pip install flask
pip install flask-socketio
pip install requests
pip install reportlab
pip install openai
pip install python-dotenv
```

### If you get permission errors:
```bash
# Make scripts executable
chmod +x five_whys_parallel.py five_whys.py run_five_whys.sh
```

### If OpenAI API fails:
- Verify your `.env` file contains the correct API key
- Check that the API key is valid and has appropriate permissions

## Dependencies Summary

Current dependencies (from requirements.txt):
- flask >= 2.3.0
- flask-socketio >= 5.3.0
- requests >= 2.31.0
- reportlab >= 4.0.0
- openai >= 1.0.0
- python-dotenv >= 1.0.0

All of these packages are compatible with Intel Macs.

## Quick Migration Checklist

- [ ] Push any uncommitted changes to git repository
- [ ] Transfer project files to Intel Mac
- [ ] Install Python 3.8+ on Intel Mac
- [ ] Create new virtual environment (`.venv`)
- [ ] Install requirements (`pip install -r requirements.txt`)
- [ ] Copy or recreate `.env` file with API keys
- [ ] Make scripts executable (`chmod +x *.py *.sh`)
- [ ] Test connection (`python3 test_connection.py`)
- [ ] Run application to verify everything works

## Notes

- The current Mac is running Python 3.13.3
- The project uses a virtual environment (`.venv`)
- Environment variables are stored in `.env` (not tracked in git)
- The `.gitignore` is properly configured to exclude architecture-specific files
