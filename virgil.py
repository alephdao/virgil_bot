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
    Generate response from Gemini model with conversation history
    """
    try:
        with model_context() as current_model:
            # Get conversation history
            history = conversation_manager.get_history(user_id)
            
            # Generate response
            if file:
                chat = current_model.start_chat(history=history)
                response = chat.send_message([input_content, file])
            else:
                chat = current_model.start_chat(history=history)
                response = chat.send_message(input_content)
            
            # Store interaction history
            conversation_manager.add_message(user_id, 'user', input_content)
            conversation_manager.add_message(user_id, 'model', response.text)
            
            return response.text
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
        "Hello! I'm your AI assistant powered by Gemini.\n\n"
        "I can:\n"
        "- Chat with you via text messages\n"
        "- Transcribe and respond to voice messages\n\n"
        "Just send me a message or audio file and I'll respond!"
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
            bot.reply_to(message, "Conversation history cleared! Let's start fresh.")
            return
            
        # Get response from Gemini
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
        bot.reply_to(message, f"Sorry, there was an error processing your message: {str(e)}")

@bot.message_handler(content_types=['voice', 'audio'])
def handle_audio(message):
    """
    Handle voice messages and audio files
    """
    try:
        # Send processing message
        processing_msg = bot.reply_to(message, "Processing your audio... Please wait.")
        
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
            "Sorry, there was an error processing your audio file.\n"
            f"Error: {str(e)}\n\n"
            "Note: telegram bots have a 20MB file limit."
        )
        bot.reply_to(message, error_message)

# Start the bot
if __name__ == "__main__":
    logger.info("Bot is running...")
    bot.infinity_polling()
