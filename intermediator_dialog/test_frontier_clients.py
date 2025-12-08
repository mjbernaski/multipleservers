#!/usr/bin/env python3
"""
Test script for frontier model clients (Anthropic, OpenAI, Gemini).
Verifies that the message handling fixes work correctly.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clients import AnthropicClient, OpenAIClient, GeminiClient
from clients import ANTHROPIC_AVAILABLE, OPENAI_AVAILABLE, GEMINI_AVAILABLE


def test_anthropic():
    """Test AnthropicClient with consecutive user messages."""
    print("\n" + "="*60)
    print("Testing Anthropic Claude (claude-opus-4.5)")
    print("="*60)

    if not ANTHROPIC_AVAILABLE:
        print("SKIP: anthropic package not installed")
        return False

    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False

    try:
        client = AnthropicClient(
            model='claude-opus-4.5',
            name='Test Claude',
            max_tokens=500
        )

        # Simulate dialog system: set system prompt via messages
        client.messages = [{"role": "system", "content": "You are a debate participant. Be concise."}]

        # Add first user message (simulating moderator intro)
        client.messages.append({"role": "user", "content": "The topic is: Should the US use its power periodically to remind the world of our strength?"})

        # Now call ask() which adds another user message - this tests the merge fix
        print("Sending test prompt (testing consecutive user message merge)...")
        response, tokens = client.ask("Please provide your opening argument in 2-3 sentences.", round_num=1, phase="test")

        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        print(f"Tokens: {tokens.get('total', 'N/A')}")
        print("SUCCESS: Anthropic client working correctly")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_openai():
    """Test OpenAIClient with consecutive user messages."""
    print("\n" + "="*60)
    print("Testing OpenAI GPT (gpt-4o)")
    print("="*60)

    if not OPENAI_AVAILABLE:
        print("SKIP: openai package not installed")
        return False

    if not os.environ.get('OPENAI_API_KEY'):
        print("SKIP: OPENAI_API_KEY not set")
        return False

    try:
        client = OpenAIClient(
            model='gpt-4o',
            name='Test GPT',
            max_tokens=500
        )

        # Simulate dialog system: set system prompt via messages
        client.messages = [{"role": "system", "content": "You are a debate participant. Be concise."}]

        # Add first user message (simulating moderator intro)
        client.messages.append({"role": "user", "content": "The topic is: Should the US use its power periodically to remind the world of our strength?"})

        # Now call ask() which adds another user message
        print("Sending test prompt (testing consecutive user message merge)...")
        response, tokens = client.ask("Please provide your opening argument in 2-3 sentences.", round_num=1, phase="test")

        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        print(f"Tokens: {tokens.get('total', 'N/A')}")
        print("SUCCESS: OpenAI client working correctly")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_gemini():
    """Test GeminiClient."""
    print("\n" + "="*60)
    print("Testing Google Gemini (gemini-2-flash)")
    print("="*60)

    if not GEMINI_AVAILABLE:
        print("SKIP: google-generativeai package not installed")
        return False

    if not os.environ.get('GOOGLE_API_KEY'):
        print("SKIP: GOOGLE_API_KEY not set")
        return False

    try:
        client = GeminiClient(
            model='gemini-2-flash',
            name='Test Gemini',
            max_tokens=500
        )

        # Simulate dialog system: set system prompt via messages
        client.messages = [{"role": "system", "content": "You are a debate participant. Be concise."}]

        # Add first user message (simulating moderator intro)
        client.messages.append({"role": "user", "content": "The topic is: Should the US use its power periodically to remind the world of our strength?"})

        # Now call ask()
        print("Sending test prompt...")
        response, tokens = client.ask("Please provide your opening argument in 2-3 sentences.", round_num=1, phase="test")

        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        print(f"Tokens: {tokens.get('total', 'N/A')}")
        print("SUCCESS: Gemini client working correctly")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("="*60)
    print("Frontier Model Client Test Suite")
    print("Testing: Anthropic Claude, OpenAI GPT, Google Gemini")
    print("="*60)

    results = {}

    results['anthropic'] = test_anthropic()
    results['openai'] = test_openai()
    results['gemini'] = test_gemini()

    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    for provider, passed in results.items():
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"  {provider.capitalize():12} : {status}")

    passed_count = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed_count}/3 providers passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
