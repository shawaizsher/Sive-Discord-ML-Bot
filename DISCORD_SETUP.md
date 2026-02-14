# Discord Bot Setup Guide

## Step 1: Create a Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **"New Application"** and give it a name (e.g., "Sive ML Bot")
3. Go to the **"Bot"** tab on the left sidebar
4. Click **"Add Bot"** and confirm
5. Under **"Privileged Gateway Intents"**, enable:
   - ✅ MESSAGE CONTENT INTENT (required for reading messages)
   - ✅ SERVER MEMBERS INTENT (optional)
6. Click **"Reset Token"** and copy the token (you'll need this!)

## Step 2: Configure Your Bot

1. Open the `.env` file in this project
2. Replace `YOUR_DISCORD_BOT_TOKEN_HERE` with your actual bot token:
   ```
   DISCORD_TOKEN=your_actual_token_here
   ```
3. **Important:** Never share your token or commit it to GitHub!

## Step 3: Invite Bot to Your Server

1. Go to **"OAuth2"** → **"URL Generator"** in the Developer Portal
2. Select scopes:
   - ✅ `bot`
   - ✅ `applications.commands`
3. Select bot permissions:
   - ✅ Read Messages/View Channels
   - ✅ Send Messages
   - ✅ Embed Links
   - ✅ Read Message History
   - ✅ Use Slash Commands
4. Copy the generated URL and open it in your browser
5. Select your server and authorize the bot

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Run the Bot

```bash
python bot.py
```

You should see:
```
Loading ML models...
All models loaded successfully!
YourBotName#1234 has connected to Discord!
```

## Available Commands

### 📊 Sentiment Analysis
```
>>analyze I love this bot!
```
Analyzes the sentiment and emotion of your text.

### 💬 AI Chatbot
```
>>chat Hello, how are you?
```
Have a conversation with the AI. It remembers your conversation history!

```
>>resetchat
```
Reset your conversation history.

### ⚠️ Content Moderation
```
>>moderate Is this message appropriate?
```
Checks if content is toxic or inappropriate.

### ✨ Text Generation
```
>>generate Once upon a time in a magical forest
```
Generates creative text from your prompt.

### ❓ Question Answering
```
>>qa Python is a programming language | What is Python?
```
Format: `>>qa <context> | <question>`

### 🤖 View All Models
```
>>models
```
Shows all available ML models and their commands.

### ℹ️ Help
```
>>help
```
Shows all available commands.

## Troubleshooting

### Bot doesn't respond
- Make sure **MESSAGE CONTENT INTENT** is enabled in Discord Developer Portal
- Check that the bot has permissions to read and send messages in the channel
- Verify your token is correct in the `.env` file

### "Models not loading"
- Run `pip install -r requirements.txt` to ensure all dependencies are installed
- Models will download automatically on first run (may take a few minutes)

### ImportError
- Make sure you're in the project directory
- Activate your virtual environment if you're using one

## Tips

- The bot uses `>>` as the command prefix
- All models are loaded once at startup for better performance
- Each user has their own chatbot conversation history
- Responses are formatted with nice embeds for better readability

## Support

For issues or questions, check the `README.md` or test models individually with `test_models.py`.
