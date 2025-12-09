#!/usr/bin/env python3
"""
N-round conversation between Claude and GPT via CLI.
Each AI gets N turns to respond to a topic/question.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()


def get_anthropic_client(model: str):
    """Create Anthropic client."""
    import anthropic
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key), model


def get_openai_client(model: str):
    """Create OpenAI client."""
    from openai import OpenAI
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    return OpenAI(api_key=api_key), model


def ask_claude(client, model: str, messages: list, system_prompt: str = None) -> str:
    """Send message to Claude and get response."""
    params = {
        "model": model,
        "max_tokens": 2048,
        "messages": messages
    }
    if system_prompt:
        params["system"] = system_prompt

    response = client.messages.create(**params)
    return response.content[0].text


def ask_gpt(client, model: str, messages: list, system_prompt: str = None) -> str:
    """Send message to GPT and get response."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(messages)

    # GPT-5.x models use max_completion_tokens, older models use max_tokens
    if model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=2048,
            messages=msgs
        )
    else:
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=msgs
        )
    return response.choices[0].message.content


def run_debate(topic: str, rounds: int, claude_model: str, gpt_model: str,
               claude_position: str = None, gpt_position: str = None):
    """Run N-round debate between Claude and GPT."""

    print("="*70)
    print("FRONTIER MODEL DEBATE")
    print("="*70)
    print(f"Topic: {topic}")
    print(f"Rounds: {rounds} (each AI responds {rounds} times)")
    print(f"Claude model: {claude_model}")
    print(f"GPT model: {gpt_model}")
    print("="*70)

    # Initialize clients
    print("\nInitializing clients...")
    claude_client, claude_model = get_anthropic_client(claude_model)
    gpt_client, gpt_model = get_openai_client(gpt_model)

    # System prompts
    claude_system = f"You are participating in a debate on the topic: {topic}"
    if claude_position:
        claude_system += f"\n\nYour position: {claude_position}"
    claude_system += "\n\nKeep responses focused and under 300 words. Engage directly with your opponent's arguments."

    gpt_system = f"You are participating in a debate on the topic: {topic}"
    if gpt_position:
        gpt_system += f"\n\nYour position: {gpt_position}"
    gpt_system += "\n\nKeep responses focused and under 300 words. Engage directly with your opponent's arguments."

    # Conversation histories (separate for each AI)
    claude_messages = []
    gpt_messages = []

    # Full transcript for display
    transcript = []

    print("\n" + "="*70)
    print("DEBATE BEGINS")
    print("="*70)

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---\n")

        # Claude's turn
        if round_num == 1:
            claude_prompt = f"The debate topic is: {topic}\n\nPlease present your opening argument."
        else:
            last_gpt = transcript[-1]['content'] if transcript else ""
            claude_prompt = f"Your opponent (GPT) said:\n\n{last_gpt}\n\nPlease respond."

        claude_messages.append({"role": "user", "content": claude_prompt})

        print(f"[Claude - {claude_model}]")
        try:
            claude_response = ask_claude(claude_client, claude_model, claude_messages, claude_system)
            print(claude_response)
            claude_messages.append({"role": "assistant", "content": claude_response})
            transcript.append({"speaker": "Claude", "content": claude_response})
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            return

        print()

        # GPT's turn
        if round_num == 1:
            gpt_prompt = f"The debate topic is: {topic}\n\nYour opponent (Claude) opened with:\n\n{claude_response}\n\nPlease respond with your position."
        else:
            gpt_prompt = f"Your opponent (Claude) said:\n\n{claude_response}\n\nPlease respond."

        gpt_messages.append({"role": "user", "content": gpt_prompt})

        print(f"[GPT - {gpt_model}]")
        try:
            gpt_response = ask_gpt(gpt_client, gpt_model, gpt_messages, gpt_system)
            print(gpt_response)
            gpt_messages.append({"role": "assistant", "content": gpt_response})
            transcript.append({"speaker": "GPT", "content": gpt_response})
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            return

    print("\n" + "="*70)
    print("DEBATE COMPLETE")
    print("="*70)
    print(f"Total exchanges: {len(transcript)}")


def interactive_mode():
    """Interactive mode where user provides input."""
    print("="*70)
    print("FRONTIER AI CHAT - Interactive Mode")
    print("="*70)

    # Get models
    claude_model = input("Claude model [claude-sonnet-4.5]: ").strip() or "claude-sonnet-4.5"
    gpt_model = input("GPT model [gpt-5.1]: ").strip() or "gpt-5.1"

    # Resolve model names
    from clients.anthropic_client import AnthropicClient
    from clients.openai_client import OpenAIClient

    claude_model = AnthropicClient.MODELS.get(claude_model, claude_model)
    gpt_model = OpenAIClient.MODELS.get(gpt_model, gpt_model)

    print(f"\nUsing Claude: {claude_model}")
    print(f"Using GPT: {gpt_model}")

    claude_client, _ = get_anthropic_client(claude_model)
    gpt_client, _ = get_openai_client(gpt_model)

    claude_messages = []
    gpt_messages = []

    print("\nCommands: 'claude <msg>', 'gpt <msg>', 'both <msg>', 'quit'")
    print("="*70)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        if user_input.lower().startswith('claude '):
            msg = user_input[7:]
            claude_messages.append({"role": "user", "content": msg})
            print(f"\n[Claude]")
            try:
                response = ask_claude(claude_client, claude_model, claude_messages)
                print(response)
                claude_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"ERROR: {e}")
                claude_messages.pop()

        elif user_input.lower().startswith('gpt '):
            msg = user_input[4:]
            gpt_messages.append({"role": "user", "content": msg})
            print(f"\n[GPT]")
            try:
                response = ask_gpt(gpt_client, gpt_model, gpt_messages)
                print(response)
                gpt_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"ERROR: {e}")
                gpt_messages.pop()

        elif user_input.lower().startswith('both '):
            msg = user_input[5:]

            claude_messages.append({"role": "user", "content": msg})
            print(f"\n[Claude]")
            try:
                c_response = ask_claude(claude_client, claude_model, claude_messages)
                print(c_response)
                claude_messages.append({"role": "assistant", "content": c_response})
            except Exception as e:
                print(f"ERROR: {e}")
                claude_messages.pop()

            gpt_messages.append({"role": "user", "content": msg})
            print(f"\n[GPT]")
            try:
                g_response = ask_gpt(gpt_client, gpt_model, gpt_messages)
                print(g_response)
                gpt_messages.append({"role": "assistant", "content": g_response})
            except Exception as e:
                print(f"ERROR: {e}")
                gpt_messages.pop()

        else:
            print("Use: 'claude <msg>', 'gpt <msg>', 'both <msg>', or 'quit'")


def main():
    parser = argparse.ArgumentParser(description="N-round debate between Claude and GPT")
    parser.add_argument('--topic', '-t', type=str, help='Debate topic')
    parser.add_argument('--rounds', '-n', type=int, default=3, help='Number of rounds (default: 3)')
    parser.add_argument('--claude-model', '-c', type=str, default='claude-sonnet-4.5',
                        help='Claude model (default: claude-sonnet-4.5)')
    parser.add_argument('--gpt-model', '-g', type=str, default='gpt-5.1',
                        help='GPT model (default: gpt-5.1)')
    parser.add_argument('--claude-position', type=str, help='Claude\'s position on the topic')
    parser.add_argument('--gpt-position', type=str, help='GPT\'s position on the topic')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.topic:
        # Prompt for topic
        topic = input("Enter debate topic: ").strip()
        if not topic:
            print("No topic provided. Use --interactive for interactive mode.")
            return
        args.topic = topic

    # Resolve model aliases
    from clients.anthropic_client import AnthropicClient
    from clients.openai_client import OpenAIClient

    claude_model = AnthropicClient.MODELS.get(args.claude_model, args.claude_model)
    gpt_model = OpenAIClient.MODELS.get(args.gpt_model, args.gpt_model)

    run_debate(
        topic=args.topic,
        rounds=args.rounds,
        claude_model=claude_model,
        gpt_model=gpt_model,
        claude_position=args.claude_position,
        gpt_position=args.gpt_position
    )


if __name__ == "__main__":
    main()
