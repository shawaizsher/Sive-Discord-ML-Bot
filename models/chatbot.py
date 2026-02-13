"""
Conversational AI Chatbot
Uses DialoGPT or BlenderBot for natural conversations
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize chatbot model
        
        Args:
            model_name: HuggingFace model name
                - "microsoft/DialoGPT-medium" (recommended, fast)
                - "microsoft/DialoGPT-large" (better quality, slower)
                - "facebook/blenderbot-400M-distill" (friendly chatbot)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Store conversation history for context
        self.chat_history_ids = None
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def respond(self, user_input, max_length=1000):
        """
        Generate a response to user input
        
        Args:
            user_input: String from user
            max_length: Maximum conversation length to track
            
        Returns:
            String response
        """
        # Encode user input and add to chat history
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        )
        
        # Append to chat history
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids
        
        # Generate response
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        # Decode response
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        return response
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.chat_history_ids = None
    
    def respond_no_history(self, user_input, max_new_tokens=100):
        """
        Generate one-time response without conversation history
        Useful for independent queries
        
        Args:
            user_input: String from user
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            String response
        """
        input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        )
        
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        return response


# Example usage
if __name__ == "__main__":
    # Create chatbot
    bot = Chatbot()
    
    # Conversation with history
    print("Chatbot: Hello! I'm your AI assistant.")
    
    # Simulate conversation
    messages = [
        "Hi! How are you?",
        "What can you help me with?",
        "Tell me a joke"
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        response = bot.respond(msg)
        print(f"Bot: {response}\n")
    
    # Reset and try without history
    bot.reset_conversation()
    response = bot.respond_no_history("What's the weather like?")
    print(f"One-time response: {response}")
