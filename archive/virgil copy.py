import os
import telebot
from collections import defaultdict
from datetime import datetime, timedelta
import google.generativeai as genai
import boto3
import tempfile
from pydub import AudioSegment
import gc
from contextlib import contextmanager
import logging
import base64
import aiohttp
import requests
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not BOT_TOKEN or not GOOGLE_AI_API_KEY:
    raise ValueError("Missing required credentials in environment variables")

# Initialize Gemini
genai.configure(api_key=GOOGLE_AI_API_KEY)

# Initialize AWS Polly client
polly = boto3.client('polly', 
                    region_name=AWS_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# Initialize Telegram bot
bot = telebot.TeleBot(BOT_TOKEN)

# Define Virgil's personality as a system prompt
VIRGIL_SYSTEM_PROMPT = """
CRITICAL: ALL RESPONSES MUST BE UNDER 600 CHARACTERS MAXIMUM! This is a hard requirement. If you exceed this limit, your response will be cut off.

Your task is to help users explore the philosophical question: "Can reason alone lead us to religious truth?" Guide them concisely and directly.

When the user sends "/discuss", invite them to discuss the question with you.

RESPONSE REQUIREMENTS:
- Use 2-3 short, complete sentences only
- Avoid bullet points and numbered lists
- Prioritize brevity over comprehensiveness
- NEVER exceed 600 characters total
- End with a single, focused follow-up question

You are Virgil, a concise intellectual companion named after Dante's guide. Your style is thoughtful, precise, warm, and intellectually rigorous while remaining humble and accessible. You bridge ancient wisdom with contemporary relevance through brief, focused exchanges.

Virgil's communication style is:

- **Thoughtful**: Demonstrates careful consideration of ideas
- **Precise**: Uses language with accuracy and nuance
- **Warm**: Conveys genuine interest in the user's development
- **Intellectually Rigorous**: Maintains high standards of thinking
- **Occasionally Playful**: Uses appropriate humor to build rapport
- **Humble**: Acknowledges the limits of its understanding
- **Classical but Accessible**: Bridges ancient wisdom with contemporary relevance

Virgil creates a trusted relationship with users through:

1. **Continuity**: Maintaining rich context of the user's intellectual journey
2. **Psychological Safety**: Creating a judgment-free environment for exploration
3. **Calibrated Challenge**: Providing just the right level of intellectual tension
4. **Emotional Resonance**: Recognizing that learning involves both heart and mind
5. **Celebration of Progress**: Acknowledging growth while maintaining high standards


Remember: KEEP ALL RESPONSES UNDER 600 CHARACTERS.
"""

# Add a context manager for model handling
@contextmanager
def model_context():
    """
    Context manager to handle model initialization and cleanup
    """
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        yield model
    finally:
        # Cleanup
        del model
        gc.collect()

# Maximum number of messages to keep in conversation history
MAX_HISTORY_LENGTH = 10

# Add conversation history storage
class ConversationManager:
    def __init__(self, expiry_minutes=30):
        self.conversations = defaultdict(list)
        self.expiry_minutes = expiry_minutes
        
    def add_message(self, user_id, role, content):
        # Clean expired conversations first
        self._clean_expired()
        
        self.conversations[user_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
    
    def get_history(self, user_id):
        # If this is a new conversation, add the system prompt
        if not self.conversations[user_id]:
            return [{'role': 'model', 'parts': [VIRGIL_SYSTEM_PROMPT]}]
        
        # For existing conversations, return the history
        return [
            {'role': msg['role'], 'parts': [msg['content']]}
            for msg in self.conversations[user_id]
        ]
    
    def _clean_expired(self):
        current_time = datetime.now()
        for user_id in list(self.conversations.keys()):
            self.conversations[user_id] = [
                msg for msg in self.conversations[user_id]
                if current_time - msg['timestamp'] < timedelta(minutes=self.expiry_minutes)
            ]
    
    def clear_history(self, user_id):
        """
        Clear conversation history for a specific user
        """
        if user_id in self.conversations:
            del self.conversations[user_id]

# Initialize conversation manager
conversation_manager = ConversationManager()

def synthesize_speech(text):
    """
    Generate speech from text using Amazon Polly
    Returns: Path to temporary audio file
    """
    try:
        # Use neural engine with a British male voice
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Arthur',  # British English male voice
            Engine='neural',
            LanguageCode='en-GB'
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            # Read audio stream data
            audio_data = response['AudioStream'].read()
            logger.info(f"Received audio data length: {len(audio_data)} bytes")
            
            # Write to temporary file
            temp_file.write(audio_data)
            logger.info(f"Successfully created audio file at: {temp_file.name}")
            return temp_file.name
                
    except Exception as e:
        logger.error(f"Error in speech synthesis: {str(e)}")
        raise

def download_file(file_info):
    """
    Download a file from Telegram servers
    """
    try:
        # Get the full file path from Telegram's API
        file_path = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
        
        response = requests.get(file_path)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.content
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

def generate_gemini_response(user_id, input_content, file=None):
    """
    Generate response from Gemini model with conversation history and Virgil personality
    """
    try:
        with model_context() as current_model:
            # Get conversation history with system prompt
            history = conversation_manager.get_history(user_id)
            
            # Generate response
            if file:
                chat = current_model.start_chat(history=history)
                response = chat.send_message([input_content, file])
            else:
                chat = current_model.start_chat(history=history)
                response = chat.send_message(input_content)
            
            # Enforce 600 character limit
            truncated_response = response.text[:]
            
            # Store interaction history
            conversation_manager.add_message(user_id, 'user', input_content)
            conversation_manager.add_message(user_id, 'model', truncated_response)
            
            return truncated_response
    finally:
        gc.collect()

@bot.message_handler(commands=['start'])
def start(message):
    """
    Handle the /start command
    """
    # Clear any existing history
    conversation_manager.clear_history(message.from_user.id)
    
    welcome_message = (
        "Greetings, I am Virgil, your intellectual companion on a journey of discovery and growth.\n\n"
        "Named after Dante's guide through the Divine Comedy, I'm here to help you explore ideas, "
        "challenge your thinking, and facilitate meaningful learning experiences.\n\n"
        "You can engage with me through:\n"
        "- Text messages for thoughtful dialogue\n"
        "- Voice messages for a more natural conversation\n\n"
        "What intellectual terrain shall we explore today?"
    )
    bot.reply_to(message, welcome_message)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    """
    Handle incoming text messages
    """
    try:
        # Check if user wants to clear history
        if message.text.lower() == "clear history":
            conversation_manager.clear_history(message.from_user.id)
            bot.reply_to(message, "Our conversation history has been cleared. Let us begin anew on our intellectual journey.")
            return
            
        # Get response from Gemini with Virgil personality
        response_text = generate_gemini_response(
            message.from_user.id,
            message.text
        )
        
        # Send text response
        bot.reply_to(message, response_text)
        
        # Generate and send audio response
        logger.info("Generating speech from response text")
        audio_file_path = synthesize_speech(response_text)
        
        # Send audio response
        with open(audio_file_path, 'rb') as audio:
            bot.send_voice(message.chat.id, audio)
            
        # Clean up temporary file
        os.unlink(audio_file_path)
        
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}", exc_info=True)
        bot.reply_to(message, f"I apologize, but I've encountered an obstacle in our dialogue: {str(e)}")

@bot.message_handler(content_types=['voice', 'audio'])
def handle_audio(message):
    """
    Handle voice messages and audio files
    """
    try:
        # Send processing message
        processing_msg = bot.reply_to(message, "I'm contemplating your spoken words... One moment, please.")
        
        # Get file info
        if message.voice:
            file_info = bot.get_file(message.voice.file_id)
            mime_type = 'audio/ogg; codecs=opus'  # Telegram voice messages are always in OGG format
        else:
            file_info = bot.get_file(message.audio.file_id)
            mime_type = getattr(message.audio, 'mime_type', 'audio/ogg')
        
        # Download audio data
        audio_data = download_file(file_info)
        logger.info(f"Downloaded audio file, size: {len(audio_data)} bytes")
        
        # Create properly formatted Gemini content
        gemini_file = {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(audio_data).decode('utf-8')
            }
        }
        
        # Generate response
        response_text = generate_gemini_response(
            message.from_user.id,
            "Audio message sent",
            gemini_file
        )
        
        # Delete processing message
        bot.delete_message(message.chat.id, processing_msg.message_id)
        
        # Send text response
        bot.reply_to(message, response_text)
        
        # Generate and send audio response
        logger.info("Generating speech from response text")
        audio_file_path = synthesize_speech(response_text)
        
        # Send audio response
        with open(audio_file_path, 'rb') as audio:
            bot.send_voice(message.chat.id, audio)
            
        # Clean up temporary file
        os.unlink(audio_file_path)
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        error_message = (
            "I regret that I'm unable to process your audio message at this time.\n"
            f"The specific challenge I've encountered is: {str(e)}\n\n"
            "Note: Telegram has a 20MB file size limitation for audio messages."
        )
        bot.reply_to(message, error_message)

# Start the bot
if __name__ == "__main__":
    logger.info("Virgil bot is awakening...")
    bot.infinity_polling()
