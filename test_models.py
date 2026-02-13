"""
Test all models to ensure they work
Run this file to test each model independently
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_analyzer import SentimentAnalyzer
from models.chatbot import Chatbot
from models.content_moderator import ContentModerator
from models.text_generator import TextGenerator
from models.qa_system import QASystem


def test_sentiment():
    print("=" * 50)
    print("TESTING SENTIMENT ANALYZER")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer(model_type="basic")
    test_texts = [
        "I love this!",
        "This is terrible",
        "It's okay, not great"
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")


def test_chatbot():
    print("=" * 50)
    print("TESTING CHATBOT")
    print("=" * 50)
    
    bot = Chatbot()
    
    messages = [
        "Hello!",
        "What can you do?",
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        response = bot.respond(msg)
        print(f"Bot: {response}\n")


def test_moderator():
    print("=" * 50)
    print("TESTING CONTENT MODERATOR")
    print("=" * 50)
    
    moderator = ContentModerator(model_type="toxic")
    
    test_messages = [
        "Hello friend!",
        "You are stupid!"
    ]
    
    for msg in test_messages:
        result = moderator.check(msg)
        print(f"Message: {msg}")
        print(f"Inappropriate: {result['is_inappropriate']}")
        print(f"Confidence: {result['confidence']:.2f}\n")


def test_generator():
    print("=" * 50)
    print("TESTING TEXT GENERATOR")
    print("=" * 50)
    
    generator = TextGenerator()
    prompt = "The future of AI is"
    
    result = generator.generate(prompt, max_length=80)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}\n")


def test_qa():
    print("=" * 50)
    print("TESTING Q&A SYSTEM")
    print("=" * 50)
    
    qa = QASystem()
    
    context = "Python is a high-level programming language. It was created by Guido van Rossum in 1991."
    question = "Who created Python?"
    
    result = qa.answer(question, context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}\n")


if __name__ == "__main__":
    print("\n🚀 Starting Model Tests...\n")
    
    try:
        test_sentiment()
        test_chatbot()
        test_moderator()
        test_generator()
        test_qa()
        
        print("=" * 50)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Make sure all packages are installed: pip install -r requirements.txt")
