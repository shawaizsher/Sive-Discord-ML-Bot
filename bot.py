"""
Sive Discord ML Bot
Discord bot with integrated Machine Learning models
"""

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

# Import ML models
from models.sentiment_analyzer import SentimentAnalyzer
from models.chatbot import Chatbot
from models.content_moderator import ContentModerator
from models.text_generator import TextGenerator
from models.qa_system import QASystem

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='>>', intents=intents, help_command=None)

# ML Models - Lazy Loading (loaded only when first used)
sentiment_analyzer = None
chatbot = None
content_moderator = None
text_generator = None
qa_system = None

# Store conversation contexts per user
user_conversations = {}


def get_sentiment_analyzer():
    """Lazy load sentiment analyzer"""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("Loading Sentiment Analyzer...")
        sentiment_analyzer = SentimentAnalyzer(model_type="basic")
        print("✓ Sentiment Analyzer loaded")
    return sentiment_analyzer


def get_content_moderator():
    """Lazy load content moderator"""
    global content_moderator
    if content_moderator is None:
        print("Loading Content Moderator...")
        content_moderator = ContentModerator(model_type="toxic")
        print("✓ Content Moderator loaded")
    return content_moderator


def get_text_generator():
    """Lazy load text generator"""
    global text_generator
    if text_generator is None:
        print("Loading Text Generator...")
        text_generator = TextGenerator()
        print("✓ Text Generator loaded")
    return text_generator


def get_qa_system():
    """Lazy load QA system"""
    global qa_system
    if qa_system is None:
        print("Loading Q&A System...")
        qa_system = QASystem()
        print("✓ Q&A System loaded")
    return qa_system


def get_user_chatbot(user_id):
    """Get or create chatbot for specific user (lazy load)"""
    if user_id not in user_conversations:
        print(f"Loading Chatbot for user {user_id}...")
        user_conversations[user_id] = Chatbot()
        print(f"✓ Chatbot loaded for user {user_id}")
    return user_conversations[user_id]


@bot.event
async def on_ready():
    print(f'✓ {bot.user} has connected to Discord!')
    print(f'✓ Bot is in {len(bot.guilds)} server(s)')
    print(f'✓ Models will load on first use (lazy loading enabled)')
    await bot.change_presence(activity=discord.Game(name=">>help for commands"))


@bot.command(name='analyze', help='Analyze sentiment of text. Usage: >>analyze <text>')
async def analyze_sentiment(ctx, *, text: str):
    """Analyze sentiment of the given text"""
    async with ctx.typing():
        analyzer = get_sentiment_analyzer()
        result = analyzer.analyze(text)
        
        # Create embed for better visualization
        embed = discord.Embed(title="📊 Sentiment Analysis", color=discord.Color.blue())
        embed.add_field(name="Text", value=text[:1000], inline=False)
        embed.add_field(name="Sentiment", value=result['label'], inline=True)
        embed.add_field(name="Confidence", value=f"{result['score']:.2%}", inline=True)
        
        await ctx.send(embed=embed)


@bot.command(name='chat', help='Chat with AI. Usage: >>chat <message>')
async def chat_with_bot(ctx, *, message: str):
    """Have a conversation with the AI chatbot"""
    async with ctx.typing():
        user_id = str(ctx.author.id)
        
        user_bot = get_user_chatbot(user_id)
        response = user_bot.respond(message)
        
        await ctx.send(response)


@bot.command(name='resetchat', help='Reset your conversation history')
async def reset_chat(ctx):
    """Reset conversation history for the user"""
    user_id = str(ctx.author.id)
    
    if user_id in user_conversations:
        user_conversations[user_id].reset_conversation()
        await ctx.send("✅ Your conversation history has been reset!")
    else:
        await ctx.send("You don't have an active conversation.")


@bot.command(name='purge', help='Delete messages from channel. Usage: >>purge <amount>')
@commands.has_permissions(manage_messages=True)
async def purge_messages(ctx, amount: int = 10):
    """Delete a specified number of messages from the channel"""
    if amount < 1 or amount > 100:
        await ctx.send("⚠️ Please specify a number between 1 and 100.")
        return
    
    try:
        # Delete the command message and the specified number of messages
        deleted = await ctx.channel.purge(limit=amount + 1)
        
        # Send confirmation message
        confirmation = await ctx.send(f"🗑️ Successfully deleted {len(deleted) - 1} message(s).")
        
        # Auto-delete confirmation after 5 seconds
        await confirmation.delete(delay=5)
        
    except discord.Forbidden:
        await ctx.send("❌ I don't have permission to delete messages in this channel.")
    except discord.HTTPException as e:
        await ctx.send(f"❌ An error occurred while deleting messages: {str(e)}")


@bot.command(name='clear', help='Clear your messages only. Usage: >>clear <amount>')
async def clear_own_messages(ctx, amount: int = 10):
    """Delete only the command user's messages"""
    if amount < 1 or amount > 100:
        await ctx.send("⚠️ Please specify a number between 1 and 100.")
        return
    
    try:
        def is_author(message):
            return message.author == ctx.author
        
        # Delete the command message and user's messages
        deleted = await ctx.channel.purge(limit=amount + 50, check=is_author)
        
        # Send confirmation message
        confirmation = await ctx.send(f"🗑️ Successfully deleted {len(deleted)} of your message(s).")
        
        # Auto-delete confirmation after 5 seconds
        await confirmation.delete(delay=5)
        
    except discord.Forbidden:
        await ctx.send("❌ I don't have permission to delete messages in this channel.")
    except discord.HTTPException as e:
        await ctx.send(f"❌ An error occurred while deleting messages: {str(e)}")


@bot.command(name='moderate', help='Check if text is inappropriate. Usage: >>moderate <text>')
async def moderate_content(ctx, *, text: str):
    """Check if content is toxic or inappropriate"""
    async with ctx.typing():
        moderator = get_content_moderator()
        result = moderator.check(text, threshold=0.7)
        
        if result['is_inappropriate']:
            embed = discord.Embed(title="⚠️ Content Moderation", color=discord.Color.red())
            embed.add_field(name="Status", value="INAPPROPRIATE", inline=True)
            embed.add_field(name="Confidence", value=f"{result['confidence']:.2%}", inline=True)
            embed.add_field(name="Reason", value="This content may be toxic or offensive", inline=False)
        else:
            embed = discord.Embed(title="✅ Content Moderation", color=discord.Color.green())
            embed.add_field(name="Status", value="APPROPRIATE", inline=True)
            embed.add_field(name="Confidence", value=f"{result['confidence']:.2%}", inline=True)
        
        await ctx.send(embed=embed)


@bot.command(name='generate', help='Generate text from prompt. Usage: >>generate <prompt>')
async def generate_text(ctx, *, prompt: str):
    """Generate creative text from a prompt"""
    async with ctx.typing():
        generator = get_text_generator()
        generated_text = generator.generate(
            prompt,
            max_length=100,
            temperature=0.8
        )
        
        embed = discord.Embed(title="✨ Text Generation", color=discord.Color.purple())
        embed.add_field(name="Prompt", value=prompt[:500], inline=False)
        embed.add_field(name="Generated Text", value=generated_text[:1000], inline=False)
        
        await ctx.send(embed=embed)


@bot.command(name='qa', help='Ask a question with context. Usage: >>qa <context> | <question>')
async def question_answer(ctx, *, text: str):
    """Answer questions based on provided context"""
    async with ctx.typing():
        # Split by | to separate context and question
        if '|' not in text:
            await ctx.send("⚠️ Please use format: `>>qa <context> | <question>`\nExample: `>>qa AI is artificial intelligence | What is AI?`")
            return
        
        parts = text.split('|', 1)
        context = parts[0].strip()
        question = parts[1].strip()
        
        qa = get_qa_system()
        answer = qa.answer(question, context)
        
        embed = discord.Embed(title="❓ Question Answering", color=discord.Color.gold())
        embed.add_field(name="Context", value=context[:500], inline=False)
        embed.add_field(name="Question", value=question, inline=False)
        embed.add_field(name="Answer", value=answer, inline=False)
        
        await ctx.send(embed=embed)


@bot.command(name='models', help='Show all available ML models')
async def show_models(ctx):
    """Display information about available models"""
    embed = discord.Embed(
        title="🤖 Available ML Models",
        description="Here are all the AI models available in this bot:",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="📊 Sentiment Analysis (>>analyze)",
        value="Detects emotions and sentiment in text",
        inline=False
    )
    
    embed.add_field(
        name="💬 Chatbot (>>chat)",
        value="Conversational AI that remembers context",
        inline=False
    )
    
    embed.add_field(
        name="⚠️ Content Moderation (>>moderate)",
        value="Detects toxic or inappropriate content",
        inline=False
    )
    
    embed.add_field(
        name="✨ Text Generation (>>generate)",
        value="Generates creative text from prompts",
        inline=False
    )
    
    embed.add_field(
        name="❓ Q&A System (>>qa)",
        value="Answers questions based on context",
        inline=False
    )
    
    await ctx.send(embed=embed)


@bot.command(name='help', help='Show all available commands')
async def help_command(ctx, command_name: str = None):
    """Custom help command with embed"""
    if command_name:
        # Show help for specific command
        command = bot.get_command(command_name)
        if command:
            embed = discord.Embed(
                title=f"ℹ️ Help: >>{command.name}",
                description=command.help or "No description available",
                color=discord.Color.blue()
            )
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"⚠️ Command `{command_name}` not found. Use `>>help` to see all commands.")
    else:
        # Show all commands
        embed = discord.Embed(
            title="📚 Sive Discord ML Bot - Commands",
            description="Here are all available commands. Use `>>help <command>` for more details.",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="📊 >>analyze <text>",
            value="Analyze sentiment and emotions in text",
            inline=False
        )
        
        embed.add_field(
            name="💬 >>chat <message>",
            value="Chat with AI that remembers your conversation",
            inline=False
        )
        
        embed.add_field(
            name="🔄 >>resetchat",
            value="Reset your conversation history",
            inline=False
        )
        
        embed.add_field(
            name="🗑️ >>purge <amount>",
            value="Delete messages from channel (requires Manage Messages permission)",
            inline=False
        )
        
        embed.add_field(
            name="🧹 >>clear <amount>",
            value="Delete only your own messages",
            inline=False
        )
        
        embed.add_field(
            name="⚠️ >>moderate <text>",
            value="Check if content is toxic or inappropriate",
            inline=False
        )
        
        embed.add_field(
            name="✨ >>generate <prompt>",
            value="Generate creative text from a prompt",
            inline=False
        )
        
        embed.add_field(
            name="❓ >>qa <context> | <question>",
            value="Answer questions based on provided context",
            inline=False
        )
        
        embed.add_field(
            name="🤖 >>models",
            value="Show all available ML models",
            inline=False
        )
        
        embed.add_field(
            name="ℹ️ >>help [command]",
            value="Show this help message or help for a specific command",
            inline=False
        )
        
        embed.set_footer(text="Powered by Hugging Face Transformers 🤗")
        
        await ctx.send(embed=embed)


@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"⚠️ Missing required argument. Use `>>help {ctx.command}` for usage info.")
    elif isinstance(error, commands.CommandNotFound):
        await ctx.send("⚠️ Command not found. Use `>>help` to see all commands.")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You don't have permission to use this command.")
    elif isinstance(error, commands.BotMissingPermissions):
        await ctx.send("❌ I don't have the required permissions to execute this command.")
    else:
        await ctx.send(f"❌ An error occurred: {str(error)}")
        print(f"Error: {error}")


# Run the bot
if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: DISCORD_TOKEN not found in .env file!")
        print("Please create a .env file with your Discord bot token.")
    else:
        bot.run(TOKEN)
