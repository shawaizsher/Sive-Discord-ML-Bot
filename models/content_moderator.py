"""
Content Moderation
Detects toxic content, hate speech, and inappropriate messages
"""

from transformers import pipeline


class ContentModerator:
    def __init__(self, model_type="toxic"):
        """
        Initialize content moderator
        
        Args:
            model_type: "toxic" for toxicity or "hate" for hate speech
        """
        if model_type == "toxic":
            self.model = pipeline(
                "text-classification",
                model="unitary/toxic-bert"
            )
        elif model_type == "hate":
            self.model = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target"
            )
        else:
            raise ValueError("model_type must be 'toxic' or 'hate'")
        
        self.model_type = model_type
    
    def check(self, text, threshold=0.7):
        """
        Check if content is inappropriate
        
        Args:
            text: String to check
            threshold: Confidence threshold (0-1). Higher = stricter
            
        Returns:
            dict with is_inappropriate (bool), label, and score
        """
        result = self.model(text)[0]
        
        # Check if toxic/hate speech
        is_inappropriate = (
            result['score'] > threshold and 
            result['label'].lower() in ['toxic', 'hate', 'label_1']
        )
        
        return {
            'is_inappropriate': is_inappropriate,
            'label': result['label'],
            'confidence': result['score']
        }
    
    def check_batch(self, texts, threshold=0.7):
        """
        Check multiple texts at once
        
        Args:
            texts: List of strings
            threshold: Confidence threshold
            
        Returns:
            List of results
        """
        results = self.model(texts)
        
        return [
            {
                'is_inappropriate': (
                    r['score'] > threshold and 
                    r['label'].lower() in ['toxic', 'hate', 'label_1']
                ),
                'label': r['label'],
                'confidence': r['score']
            }
            for r in results
        ]


# Example usage
if __name__ == "__main__":
    # Toxic content detector
    moderator = ContentModerator(model_type="toxic")
    
    test_messages = [
        "Hello! How are you today?",
        "You're stupid and useless!",
        "This is a normal message"
    ]
    
    for msg in test_messages:
        result = moderator.check(msg)
        print(f"Message: {msg}")
        print(f"Inappropriate: {result['is_inappropriate']}")
        print(f"Confidence: {result['confidence']:.2f}\n")
