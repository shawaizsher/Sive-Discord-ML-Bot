"""
Text Generation
Generate creative text, stories, or completions
"""

from transformers import pipeline


class TextGenerator:
    def __init__(self, model_name="gpt2"):
        """
        Initialize text generator
        
        Args:
            model_name: HuggingFace model name
                - "gpt2" (fast, good quality)
                - "gpt2-medium" (better quality)
                - "EleutherAI/gpt-neo-1.3B" (high quality, needs more RAM)
        """
        self.generator = pipeline(
            "text-generation",
            model=model_name
        )
        self.model_name = model_name
    
    def generate(self, prompt, max_length=100, num_return=1, temperature=0.8):
        """
        Generate text from a prompt
        
        Args:
            prompt: Starting text
            max_length: Maximum total length (prompt + generated)
            num_return: Number of different generations to return
            temperature: Creativity (0.1=conservative, 1.5=creative)
            
        Returns:
            Generated text or list of texts
        """
        results = self.generator(
            prompt,
            max_new_tokens=max_length,
            num_return_sequences=num_return,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        if num_return == 1:
            return results[0]['generated_text']
        else:
            return [r['generated_text'] for r in results]
    
    def complete_sentence(self, text, max_new_tokens=50):
        """
        Complete an incomplete sentence
        
        Args:
            text: Incomplete text
            max_new_tokens: Max tokens to add
            
        Returns:
            Completed text
        """
        result = self.generator(
            text,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        return result[0]['generated_text']


# Example usage
if __name__ == "__main__":
    # Create generator
    generator = TextGenerator()
    
    # Generate a story
    prompt = "Once upon a time, in a magical forest,"
    story = generator.generate(prompt, max_length=150)
    print(f"Generated Story:\n{story}\n")
    
    # Multiple variations
    variations = generator.generate(
        "The best thing about AI is",
        max_length=50,
        num_return=3
    )
    
    print("Three variations:")
    for i, text in enumerate(variations, 1):
        print(f"{i}. {text}\n")
    
    # Complete sentence
    incomplete = "The weather today is"
    completed = generator.complete_sentence(incomplete)
    print(f"Completion: {completed}")
