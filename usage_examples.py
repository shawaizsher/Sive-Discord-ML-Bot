"""
Quick usage examples for all models
Shows how to import and use each model in your own code
"""

# ============================================
# 1. SENTIMENT ANALYSIS
# ============================================

from models.sentiment_analyzer import SentimentAnalyzer

# Initialize
sentiment = SentimentAnalyzer(model_type="basic")  # or "social" or "emotions"

# Single analysis
result = sentiment.analyze("I love this bot!")
print(result)  # {'label': 'POSITIVE', 'score': 0.9998}

# Batch analysis (faster for multiple texts)
results = sentiment.analyze_batch([
    "Great job!",
    "This is terrible",
    "It's okay"
])


# ============================================
# 2. CHATBOT (Conversational AI)
# ============================================

from models.chatbot import Chatbot

# Initialize
bot = Chatbot()  # Uses DialoGPT-medium by default

# Response with conversation history (remembers context)
response = bot.respond("Hello! How are you?")

# Continue conversation
response = bot.respond("Tell me a joke")

# Reset conversation
bot.reset_conversation()

# One-time response (no history)
response = bot.respond_no_history("What's the weather?")


# ============================================
# 3. CONTENT MODERATION
# ============================================

from models.content_moderator import ContentModerator

# Initialize
moderator = ContentModerator(model_type="toxic")  # or "hate"

# Check single message
result = moderator.check("You're stupid!", threshold=0.7)
if result['is_inappropriate']:
    print(f"Blocked! Confidence: {result['confidence']}")

# Check multiple messages
results = moderator.check_batch([
    "Hello friend",
    "You suck!",
    "Nice work"
])


# ============================================
# 4. TEXT GENERATION
# ============================================

from models.text_generator import TextGenerator

# Initialize
generator = TextGenerator()  # Uses GPT-2 by default

# Generate text
text = generator.generate(
    "Once upon a time",
    max_length=100,
    temperature=0.8  # 0.1=conservative, 1.5=creative
)

# Multiple variations
variations = generator.generate(
    "The best thing is",
    num_return=3
)

# Complete sentence
completed = generator.complete_sentence("Today I will")


# ============================================
# 5. QUESTION ANSWERING
# ============================================

from models.qa_system import QASystem

# Initialize
qa = QASystem()

# Answer from context
context = "The Eiffel Tower is in Paris. It was built in 1889."
result = qa.answer("Where is the Eiffel Tower?", context)
print(result['answer'])  # "Paris"

# Multiple questions
questions = [
    "Where is the Eiffel Tower?",
    "When was it built?"
]
answers = qa.answer_multiple(questions, context)


# ============================================
# PRACTICAL EXAMPLE: Discord Message Handler
# ============================================

def handle_message(message_content):
    """Process a Discord message with ML"""
    
    # 1. Check if inappropriate
    moderator = ContentModerator()
    mod_result = moderator.check(message_content)
    
    if mod_result['is_inappropriate']:
        return "⚠️ Message blocked for inappropriate content"
    
    # 2. Analyze sentiment
    sentiment = SentimentAnalyzer()
    sentiment_result = sentiment.analyze(message_content)
    
    # 3. Generate response
    bot = Chatbot()
    response = bot.respond(message_content)
    
    return {
        'response': response,
        'sentiment': sentiment_result['label'],
        'is_safe': True
    }


# Example usage
if __name__ == "__main__":
    # Test the handler
    result = handle_message("Hello! How are you today?")
    print(result)
