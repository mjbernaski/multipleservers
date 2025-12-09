#!/usr/bin/env python3
"""
Simple chat interface to test frontier AI models (Claude, GPT).
This isolates the cloud API integration from the dialog system.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_anthropic():
    """Test Anthropic Claude API directly."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return False

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment")
        return False

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    client = anthropic.Anthropic(api_key=api_key)

    # Test with a real model that exists
    test_models = [
        'claude-sonnet-4-20250514',
        'claude-3-5-sonnet-20241022',
        'claude-3-haiku-20240307',
    ]

    for model in test_models:
        print(f"\nTesting model: {model}")
        try:
            response = client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Say hello in one sentence."}]
            )
            print(f"  SUCCESS: {response.content[0].text}")
            return model  # Return the working model
        except anthropic.NotFoundError as e:
            print(f"  Model not found: {e}")
        except anthropic.AuthenticationError as e:
            print(f"  Auth error: {e}")
            return False
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

    print("\nNo working Anthropic models found!")
    return False


def test_openai():
    """Test OpenAI API directly."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        return False

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment")
        return False

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    client = OpenAI(api_key=api_key)

    # Test with real models that exist
    test_models = [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4-turbo',
        'gpt-3.5-turbo',
    ]

    for model in test_models:
        print(f"\nTesting model: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Say hello in one sentence."}]
            )
            print(f"  SUCCESS: {response.choices[0].message.content}")
            return model  # Return the working model
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

    print("\nNo working OpenAI models found!")
    return False


def chat_with_anthropic(model: str):
    """Interactive chat with Claude."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    messages = []

    print(f"\n{'='*60}")
    print(f"Chatting with Claude ({model})")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print('='*60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=messages
            )
            assistant_msg = response.content[0].text
            messages.append({"role": "assistant", "content": assistant_msg})
            print(f"\nClaude: {assistant_msg}")
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}")
            messages.pop()  # Remove failed user message


def chat_with_openai(model: str):
    """Interactive chat with GPT."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    messages = []

    print(f"\n{'='*60}")
    print(f"Chatting with GPT ({model})")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print('='*60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=messages
            )
            assistant_msg = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_msg})
            print(f"\nGPT: {assistant_msg}")
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}")
            messages.pop()  # Remove failed user message


def main():
    print("="*60)
    print("FRONTIER MODEL TEST")
    print("="*60)

    print("\n[1] Testing Anthropic (Claude)...")
    anthropic_model = test_anthropic()

    print("\n" + "-"*60)

    print("\n[2] Testing OpenAI (GPT)...")
    openai_model = test_openai()

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Anthropic: {'WORKING - ' + anthropic_model if anthropic_model else 'NOT WORKING'}")
    print(f"OpenAI:    {'WORKING - ' + openai_model if openai_model else 'NOT WORKING'}")

    if not anthropic_model and not openai_model:
        print("\nBoth providers failed. Check your API keys and network connection.")
        return

    # Offer chat
    print("\n" + "="*60)
    print("INTERACTIVE CHAT")
    print("="*60)

    options = []
    if anthropic_model:
        options.append(('1', 'Claude', lambda: chat_with_anthropic(anthropic_model)))
    if openai_model:
        options.append(('2', 'GPT', lambda: chat_with_openai(openai_model)))
    options.append(('q', 'Quit', None))

    while True:
        print("\nSelect an AI to chat with:")
        for key, name, _ in options:
            print(f"  [{key}] {name}")

        try:
            choice = input("\nChoice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        for key, name, func in options:
            if choice == key:
                if func:
                    func()
                else:
                    print("Goodbye!")
                    return
                break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
