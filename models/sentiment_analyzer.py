"""
Sentiment Analysis Model
Uses distilbert for basic sentiment or RoBERTa for emotion detection
"""

from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self, model_type="basic"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_type: "basic" for positive/negative or "emotions" for 28 emotions
        """
        if model_type == "basic":
            # Fast, simple positive/negative sentiment
            self.model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        elif model_type == "social":
            # Better for social media/Twitter-like text
            self.model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment"
            )
        elif model_type == "emotions":
            # Detects 28 different emotions
            self.model = pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=None
            )
        else:
            raise ValueError("model_type must be 'basic', 'social', or 'emotions'")
        
        self.model_type = model_type
    
    def analyze(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text: String to analyze
            
        Returns:
            dict with label and score
        """
        result = self.model(text)
        
        if self.model_type == "emotions":
            # Return top 3 emotions
            return result[0][:3]
        else:
            return result[0]
    
    def analyze_batch(self, texts):
        """
        Analyze multiple texts at once (faster)
        
        Args:
            texts: List of strings
            
        Returns:
            List of results
        """
        return self.model(texts)


# Example usage
if __name__ == "__main__":
    # Basic sentiment
    analyzer = SentimentAnalyzer(model_type="basic")
    result = analyzer.analyze("I love this bot! It's amazing!")
    print(f"Basic Sentiment: {result}")
    
    # Social media sentiment
    social_analyzer = SentimentAnalyzer(model_type="social")
    result = social_analyzer.analyze("This is so cool! 🎉")
    print(f"Social Sentiment: {result}")
    
    # Emotions
    emotion_analyzer = SentimentAnalyzer(model_type="emotions")
    result = emotion_analyzer.analyze("I'm so excited and happy!")
    print(f"Emotions: {result}")
