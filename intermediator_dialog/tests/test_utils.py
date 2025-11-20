"""
Unit tests for utility functions.
"""
import unittest
from unittest.mock import patch, mock_open
from datetime import datetime
from utils import generate_filename_from_topic, debug_log


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_generate_filename_from_topic_simple(self):
        """Test filename generation from simple topic."""
        topic = "What is artificial intelligence?"
        filename = generate_filename_from_topic(topic)

        self.assertIn("What_is_artificial_intelligence", filename)
        self.assertTrue(len(filename) > 20)  # Should include timestamp

    def test_generate_filename_from_topic_long(self):
        """Test filename generation from long topic."""
        topic = "A" * 200  # Very long topic
        filename = generate_filename_from_topic(topic, max_length=60)

        # Should be truncated
        self.assertLessEqual(len(filename.split('_202')[0]), 60)

    def test_generate_filename_from_topic_special_chars(self):
        """Test filename generation removes special characters."""
        topic = "What's the #1 best approach to AI? @OpenAI says..."
        filename = generate_filename_from_topic(topic)

        # Should not contain special characters
        self.assertNotIn('#', filename)
        self.assertNotIn('@', filename)
        self.assertNotIn('?', filename)

    def test_generate_filename_from_topic_empty(self):
        """Test filename generation from empty topic."""
        filename = generate_filename_from_topic("")

        self.assertTrue(filename.startswith("dialog_"))

    def test_generate_filename_from_topic_multiline(self):
        """Test filename generation from multiline topic."""
        topic = "First line\nSecond line\nThird line"
        filename = generate_filename_from_topic(topic)

        # Should only use first line
        self.assertIn("First_line", filename)
        self.assertNotIn("Second", filename)

    def test_debug_log_without_socketio(self):
        """Test debug_log without SocketIO."""
        with patch('builtins.print') as mock_print:
            debug_log('info', 'Test message')

            # Should call print
            mock_print.assert_called_once()
            call_args = str(mock_print.call_args)
            self.assertIn('INFO', call_args)
            self.assertIn('Test message', call_args)

    def test_debug_log_with_server(self):
        """Test debug_log with server parameter."""
        with patch('builtins.print') as mock_print:
            debug_log('error', 'Error occurred', server='Server1')

            call_args = str(mock_print.call_args)
            self.assertIn('ERROR', call_args)
            self.assertIn('Server1', call_args)
            self.assertIn('Error occurred', call_args)


if __name__ == '__main__':
    unittest.main()
