"""
Integration tests for the dialog system.
"""
import unittest
from unittest.mock import Mock, patch
from clients.ollama_client import OllamaClient
from clients.base_client import BaseClient
from intermediator_dialog_refactored import IntermediatorDialogRefactored, DialogConfig
from prompt_templates import DialogMode


class MockClient(BaseClient):
    """Mock client for testing."""

    def __init__(self, model: str, name: str):
        super().__init__(model=model, name=name)
        self.ask_count = 0
        self.system_prompt_value = None

    def ask(self, question: str, round_num: int = 0, phase: str = None):
        """Mock ask method."""
        self.ask_count += 1
        response = f"Response {self.ask_count} from {self.name}"
        token_info = {
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total': 30,
            'tokens_per_second': 15.0,
            'time_to_first_token': 0.1
        }
        self.messages.append({'role': 'assistant', 'content': response})
        return response, token_info

    def check_server_available(self):
        """Mock server check."""
        return True

    def reset_conversation(self):
        """Mock reset."""
        self.messages = []
        self.total_tokens = 0

    def preload_model(self):
        """Mock preload."""
        return True

    def set_system_prompt(self, prompt: str):
        """Mock set system prompt."""
        self.system_prompt_value = prompt
        self.messages = [{"role": "system", "content": prompt}]


class TestIntegration(unittest.TestCase):
    """Integration tests for dialog system."""

    def test_dialog_flow_basic(self):
        """Test basic dialog flow with mock clients."""
        intermediator = MockClient("test-model", "Intermediator")
        participant1 = MockClient("test-model", "Participant1")
        participant2 = MockClient("test-model", "Participant2")

        config = DialogConfig(
            mode=DialogMode.DEBATE,
            max_turns=1,
            enable_tts=False,
        )

        dialog = IntermediatorDialogRefactored(
            intermediator=intermediator,
            participant1=participant1,
            participant2=participant2,
            topic="Discuss AI safety",
            config=config,
        )

        result = dialog.run_dialog()

        self.assertIn('conversation_history', result)
        self.assertIn('runtime_seconds', result)
        self.assertIn('total_turns', result)

        history = result['conversation_history']
        self.assertGreater(len(history), 0)

        participant_speakers = [msg['speaker'] for msg in history]
        self.assertIn('intermediator', participant_speakers)
        self.assertIn('participant1', participant_speakers)
        self.assertIn('participant2', participant_speakers)

        self.assertGreater(intermediator.ask_count, 0)
        self.assertGreater(participant1.ask_count, 0)
        self.assertGreater(participant2.ask_count, 0)

    def test_dialog_system_prompts(self):
        """Test that system prompts are set correctly."""
        intermediator = MockClient("test", "Int")
        participant1 = MockClient("test", "P1")
        participant2 = MockClient("test", "P2")

        config = DialogConfig(
            mode=DialogMode.DEBATE,
            max_turns=1,
            enable_tts=False,
            moderator_instructions="You are a moderator.",
            participant1_position="You represent viewpoint A.",
            participant2_position="You represent viewpoint B.",
        )

        dialog = IntermediatorDialogRefactored(
            intermediator=intermediator,
            participant1=participant1,
            participant2=participant2,
            topic="Topic: AI Ethics",
            config=config,
        )

        dialog.run_dialog()

        self.assertGreater(len(intermediator.messages), 0)
        self.assertGreater(len(participant1.messages), 0)
        self.assertGreater(len(participant2.messages), 0)

        self.assertEqual(intermediator.messages[0]['role'], 'system')
        self.assertEqual(participant1.messages[0]['role'], 'system')
        self.assertEqual(participant2.messages[0]['role'], 'system')

    def test_base_client_interface(self):
        """Test that BaseClient interface is properly defined."""
        from clients.base_client import BaseClient

        self.assertTrue(hasattr(BaseClient, 'ask'))
        self.assertTrue(hasattr(BaseClient, 'check_server_available'))
        self.assertTrue(hasattr(BaseClient, 'reset_conversation'))

        self.assertTrue(hasattr(BaseClient, 'add_message'))
        self.assertTrue(hasattr(BaseClient, 'get_conversation_length'))
        self.assertTrue(hasattr(BaseClient, 'get_total_tokens'))

    def test_ollama_client_inherits_base(self):
        """Test that OllamaClient properly inherits from BaseClient."""
        self.assertTrue(issubclass(OllamaClient, BaseClient))

        client = OllamaClient(
            host="http://localhost:11434",
            model="test",
            name="Test"
        )

        self.assertTrue(hasattr(client, 'add_message'))
        self.assertTrue(hasattr(client, 'get_conversation_length'))
        self.assertTrue(hasattr(client, 'set_stream_callback'))


if __name__ == '__main__':
    unittest.main()
