import unittest
from unittest.mock import MagicMock, patch
from intermediator_dialog_refactored import IntermediatorDialogRefactored, DialogConfig, DialogMode
from prompt_templates import PromptTemplates

class MockOllamaClient:
    def __init__(self, name):
        self.name = name
        self.messages = []
        self.thinking = False
        self.stream_callback = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def set_stream_callback(self, callback):
        self.stream_callback = callback
    
    def preload_model(self):
        pass

    def ask(self, prompt, round_num=0, phase=None):
        # Simulate response based on phase
        if phase == "draft":
            return f"Draft response from {self.name}", {}
        elif phase == "critique":
            return f"Critique from {self.name}", {}
        elif phase == "final":
            return f"Final response from {self.name}", {}
        else:
            # Standard ask (e.g. moderator or intro)
            return f"Response from {self.name} to prompt", {}

class TestIntermediatorDialogRefactored(unittest.TestCase):
    def setUp(self):
        self.intermediator = MockOllamaClient("Moderator")
        self.participant1 = MockOllamaClient("Participant1")
        self.participant2 = MockOllamaClient("Participant2")
        
        self.config = DialogConfig(
            mode=DialogMode.DEBATE,
            max_turns=1, # Short test
            enable_tts=False
        )
        
        self.dialog = IntermediatorDialogRefactored(
            intermediator=self.intermediator,
            participant1=self.participant1,
            participant2=self.participant2,
            topic="Test Topic",
            config=self.config
        )

    def test_handle_participant_turn_flow(self):
        # We want to verify that _handle_participant_turn calls ask 3 times with correct phases
        
        # Spy on participant1.ask
        self.participant1.ask = MagicMock(wraps=self.participant1.ask)
        
        # Run a single turn manually or via run_dialog
        print("\n--- Starting Dialog Test ---")
        result = self.dialog.run_dialog()
        print("--- Dialog Test Complete ---")
        
        # Verify participant 1 was called with phases
        p1_calls = self.participant1.ask.call_args_list
        
        draft_call = False
        critique_call = False
        final_call = False
        
        for args, kwargs in p1_calls:
            phase = kwargs.get('phase')
            if phase == 'draft':
                draft_call = True
            elif phase == 'critique':
                critique_call = True
            elif phase == 'final':
                final_call = True
                
        self.assertTrue(draft_call, "Participant 1 should have a draft phase")
        self.assertTrue(critique_call, "Participant 1 should have a critique phase")
        self.assertTrue(final_call, "Participant 1 should have a final phase")
        
        # Verify history contains the FINAL response
        p1_history = [m for m in result['conversation_history'] if m['speaker'] == 'participant1']
        self.assertTrue(len(p1_history) > 0)
        self.assertIn("Final response", p1_history[0]['message'])

if __name__ == '__main__':
    unittest.main()
