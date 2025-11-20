"""
Unit tests for OllamaClient.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from clients.ollama_client import OllamaClient


class TestOllamaClient(unittest.TestCase):
    """Test cases for OllamaClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = OllamaClient(
            host="http://localhost:11434",
            model="test-model",
            name="Test Client",
            num_ctx=8192,
            temperature=0.7
        )

    def test_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.host, "http://localhost:11434")
        self.assertEqual(self.client.model, "test-model")
        self.assertEqual(self.client.name, "Test Client")
        self.assertEqual(self.client.num_ctx, 8192)
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.messages, [])
        self.assertEqual(self.client.total_tokens, 0)

    def test_host_trailing_slash_removal(self):
        """Test that trailing slashes are removed from host."""
        client = OllamaClient(
            host="http://localhost:11434/",
            model="test",
            name="Test"
        )
        self.assertEqual(client.host, "http://localhost:11434")

    @patch('clients.ollama_client.requests.get')
    def test_check_server_available_success(self, mock_get):
        """Test successful server availability check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'test-model'},
                {'name': 'other-model'}
            ]
        }
        mock_get.return_value = mock_response

        result = self.client.check_server_available()

        self.assertTrue(result)
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=5
        )

    @patch('clients.ollama_client.requests.get')
    def test_check_server_available_model_not_found(self, mock_get):
        """Test server check when model is not available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'other-model'}
            ]
        }
        mock_get.return_value = mock_response

        result = self.client.check_server_available()

        self.assertFalse(result)

    @patch('clients.ollama_client.requests.get')
    def test_check_server_available_connection_error(self, mock_get):
        """Test server check with connection error."""
        mock_get.side_effect = Exception("Connection refused")

        result = self.client.check_server_available()

        self.assertFalse(result)

    @patch('clients.ollama_client.requests.post')
    def test_ask_success(self, mock_post):
        """Test successful ask request."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({
                'message': {'content': 'Hello'},
                'done': False
            }).encode(),
            json.dumps({
                'message': {'content': ' world'},
                'done': False
            }).encode(),
            json.dumps({
                'done': True,
                'prompt_eval_count': 10,
                'eval_count': 5,
                'eval_duration': 1000000000  # 1 second in nanoseconds
            }).encode()
        ]
        mock_post.return_value = mock_response

        answer, token_info = self.client.ask("Test question")

        self.assertEqual(answer, "Hello world")
        self.assertEqual(token_info['prompt_tokens'], 10)
        self.assertEqual(token_info['completion_tokens'], 5)
        self.assertEqual(token_info['total'], 15)
        self.assertAlmostEqual(token_info['tokens_per_second'], 5.0, places=1)

        # Check message history
        self.assertEqual(len(self.client.messages), 2)
        self.assertEqual(self.client.messages[0]['role'], 'user')
        self.assertEqual(self.client.messages[0]['content'], 'Test question')
        self.assertEqual(self.client.messages[1]['role'], 'assistant')
        self.assertEqual(self.client.messages[1]['content'], 'Hello world')

    @patch('clients.ollama_client.requests.post')
    def test_ask_with_stream_callback(self, mock_post):
        """Test ask with streaming callback."""
        callback_data = []

        def mock_callback(data):
            callback_data.append(data)

        self.client.set_stream_callback(mock_callback)

        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({
                'message': {'content': 'Test'},
                'done': False
            }).encode(),
            json.dumps({
                'done': True,
                'prompt_eval_count': 5,
                'eval_count': 3,
                'eval_duration': 1000000000
            }).encode()
        ]
        mock_post.return_value = mock_response

        answer, token_info = self.client.ask("Question")

        # Check callbacks were called
        self.assertEqual(len(callback_data), 2)
        self.assertEqual(callback_data[0]['type'], 'content')
        self.assertEqual(callback_data[0]['content'], 'Test')
        self.assertEqual(callback_data[1]['type'], 'response_complete')

    def test_reset_conversation(self):
        """Test conversation reset."""
        self.client.messages = [{'role': 'user', 'content': 'test'}]
        self.client.total_tokens = 100
        self.client.total_prompt_tokens = 50
        self.client.total_completion_tokens = 50

        self.client.reset_conversation()

        self.assertEqual(self.client.messages, [])
        self.assertEqual(self.client.total_tokens, 0)
        self.assertEqual(self.client.total_prompt_tokens, 0)
        self.assertEqual(self.client.total_completion_tokens, 0)

    @patch('clients.ollama_client.requests.post')
    def test_preload_model_success(self, mock_post):
        """Test successful model preloading."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = self.client.preload_model()

        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch('clients.ollama_client.requests.post')
    def test_preload_model_failure(self, mock_post):
        """Test model preloading failure."""
        mock_post.side_effect = Exception("Connection error")

        result = self.client.preload_model()

        self.assertFalse(result)

    def test_be_brief_prepends_text(self):
        """Test that be_brief prepends text to questions."""
        client = OllamaClient(
            host="http://localhost:11434",
            model="test",
            name="Test",
            be_brief=True
        )

        with patch('clients.ollama_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                json.dumps({'done': True, 'prompt_eval_count': 0, 'eval_count': 0, 'eval_duration': 1}).encode()
            ]
            mock_post.return_value = mock_response

            client.ask("Hello")

            # Check that "Be Brief. " was prepended
            self.assertEqual(client.messages[0]['content'], "Be Brief. Hello")


if __name__ == '__main__':
    unittest.main()
