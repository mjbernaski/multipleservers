#!/usr/bin/env python3
"""Test connection to Ollama server"""
import requests

host = "http://192.168.5.40:11434"
url = f"{host}/api/tags"

print(f"Testing connection to: {url}")

try:
    response = requests.get(url, timeout=5)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("\n✓ Connection successful!")
except requests.exceptions.ConnectionError as e:
    print(f"✗ Connection Error: {e}")
except requests.exceptions.Timeout as e:
    print(f"✗ Timeout Error: {e}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
