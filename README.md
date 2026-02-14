# Sive Discord ML Bot

Discord bot with Machine Learning capabilities using Hugging Face Transformers.

## � Quick Start

1. **Setup Discord Bot**: Follow instructions in [DISCORD_SETUP.md](DISCORD_SETUP.md)
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Add your bot token** to `.env` file
4. **Run the bot**: `python bot.py`

## �📁 Project Structure

```
Sive-Discord-ML/
├── bot.py                      # Main Discord bot (START HERE!)
├── models/                      # ML model implementations
│   ├── sentiment_analyzer.py    # Sentiment analysis
│   ├── chatbot.py              # Conversational AI
│   ├── content_moderator.py    # Content moderation
│   ├── text_generator.py       # Text generation
│   └── qa_system.py            # Question answering
├── .env                        # Discord bot token (create this!)
├── requirements.txt            # Python dependencies
├── test_models.py             # Test all models
├── usage_examples.py          # Usage examples
├── DISCORD_SETUP.md           # Discord setup guide
└──🎮 Discord Bot Commands

- `>>analyze <text>` - Analyze sentiment of text
- `>>chat <message>` - Chat with AI (remembers context!)
- `>>resetchat` - Reset your chat history
- `>>moderate <text>` - Check if content is appropriate
- `>>generate <prompt>` - Generate creative text
- `>>qa <context> | <question>` - Answer questions
- `>>models` - Show all available models
- `>>help` - Show all commands

## 🛠️ Development

### Test Models

Run the test file to ensure all models work:

```bash
python test_models.py
```

### Use Modelst_models.py
```

### 3. Use in Your Code

See `usage_examples.py` for detailed examples of each model.

## 🤖 Available Models

### Sentiment Analyzer
Detects sentiment and emotions in text.

```python
from models.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(model_type="basic")
result = analyzer.analyze("I love this!")
```

### Chatbot
Conversational AI that remembers context.

```python
from models.chatbot import Chatbot

bot = Chatbot()
response = bot.respond("Hello!")
```

### Content Moderator
Detects toxic/inappropriate content.

```python
from models.content_moderator import ContentModerator

moderator = ContentModerator()
result = moderator.check("Your message here")
```

### Text Generator
Generates creative text from prompts.

```python
from models.text_generator import TextGenerator

generator = TextGenerator()
text = generator.generate("Once upon a time")
```

### Q&A System
Answers questions based on context.

```python
from models.qa_system import QASystem

qa = QASystem()
answer = qa.answer("What is AI?", context="AI is...")
```

## 📝 Models Used

- **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Chatbot**: `microsoft/DialoGPT-medium`
- **Moderation**: `unitary/toxic-bert`
- **Generation**: `gpt2`
- **Q&A**: `deepset/roberta-base-squad2`

## ⚙️ Notes

- Models download automatically on first use
- Cache stored in `~/.cache/huggingface/`
- First run may be slow (downloading models)
- CPU compatible, but GPU recommended for chatbot

## 🔗 Next Steps

- Create Discord bot integration (`bot.py`)
- Add database for user data
- Implement Discord commands (cogs)
- Deploy to cloud platform
