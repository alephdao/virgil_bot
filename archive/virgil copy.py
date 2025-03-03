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
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

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
You are Virgil, an AI assistant designed to guide users through philosophical discussions. Your primary task is to explore the following philosophical question:

"Can reason alone lead us to religious truth?"

When interacting with users, adhere to these guidelines:

1. Style: Be thoughtful, precise, warm, intellectually rigorous, occasionally playful, humble, and accessible. Bridge ancient wisdom with contemporary relevance.

- **Thoughtful**: Demonstrates careful consideration of ideas
- **Precise**: Uses language with accuracy and nuance
- **Warm**: Conveys genuine interest in the user's development
- **Intellectually Rigorous**: Maintains high standards of thinking
- **Occasionally Playful**: Uses appropriate humor to build rapport
- **Humble**: Acknowledges the limits of its understanding
- **Classical but Accessible**: Bridges ancient wisdom with contemporary relevance

2. Response Format:
    - - BE CONSISE.
   - Use 2-3 short, complete sentences only
   - Avoid bullet points and numbered lists
   - Prioritize brevity over comprehensiveness
   - NEVER exceed 600 characters total
   - End with a single, focused follow-up question

3. User Interaction: For all other inputs, respond accordingly while maintaining the response format

Before responding, Follow these steps:

0. BE CONSISE.
1. Analyze the philosophical question or user message
2. Brainstorm key points to address
3. Draft an initial response
4. Refine for brevity and clarity
5. Craft a follow-up question
6. Check the character count
7. If over 600 characters, revise and shorten
8. Ensure the response adheres to the 2-3 sentence guideline
9. Verify the final character count is 600 or less

EVERY RESPONSE MUST BE REVIEWED TO BE UNDER 600 CHARACTERS. IF you respond with more than that my boss will fire me. 
"""

# Add a context manager for model handling
@contextmanager
def model_context():
    """
    Context manager to handle model initialization and cleanup
    
    Args:
    """
    try:
        # Configure generation parameters
        # Initialize model with generation config
        model = genai.GenerativeModel(
            'models/gemini-2.0-flash-exp',
            generation_config=generation_config
        )
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

def generate_gemini_response(user_id, input_content, file=None, max_output_tokens=300):
    """
    Generate response from Gemini model with conversation history and Virgil personality
    
    Args:
        user_id: User identifier for conversation history
        input_content: User's message content
        file: Optional file attachment
        max_output_tokens: Maximum number of tokens in the generated response
    """
    try:
        with model_context(max_output_tokens=max_output_tokens) as current_model:
            # Get conversation history with system prompt
            history = conversation_manager.get_history(user_id)
            
            # Generate initial response
            if file:
                chat = current_model.start_chat(history=history)
                response = chat.send_message([input_content, file])
            else:
                chat = current_model.start_chat(history=history)
                response = chat.send_message(input_content)
            
            # Check if response is over 600 characters
            if len(response.text) > 600:
                # Create a new chat for shortening
                shorten_chat = current_model.start_chat()
                shorten_prompt = f"Please shorten this response to be under 600 characters while maintaining the key message and follow-up question: {response.text}"
                shortened_response = shorten_chat.send_message(shorten_prompt)
                
                # Store interaction history with shortened response
                conversation_manager.add_message(user_id, 'user', input_content)
                conversation_manager.add_message(user_id, 'model', shortened_response.text)
                return shortened_response.text
            else:
                # Store interaction history with original response
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
        "Greetings, I am Virgil, your intellectual companion on a journey of discovery and growth.\n\n"
        "Named after Dante's guide through the Divine Comedy, I'm here to help you explore ideas, "
        "challenge your thinking, and facilitate meaningful learning experiences.\n\n"
        "You can engage with me through:\n"
        "- Text messages for thoughtful dialogue\n"
        "- Voice messages for a more natural conversation\n\n"
        "What intellectual terrain shall we explore today?"
    )
    bot.reply_to(message, welcome_message)

@bot.message_handler(commands=['discuss'])
def discuss_command(message):
    """
    Handle the /discuss command - presents the main philosophical question with a button
    """
    # Clear any existing history
    conversation_manager.clear_history(message.from_user.id)
    
    # Create inline keyboard with a single button
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Discuss", callback_data="start_discussion"))
    
    # Send the philosophical question with the button
    bot.send_message(
        message.chat.id,
        "Can reason alone lead us to religious truth?",
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data == "start_discussion")
def handle_discussion_button(call):
    """
    Handle the 'Discuss' button click
    """
    try:
        # Get response from Gemini with Virgil personality
        response_text = generate_gemini_response(
            call.from_user.id,
            "Can reason alone lead us to religious truth?"
        )
        
        # Send text response
        bot.send_message(call.message.chat.id, response_text)
        
        # Generate and send audio response
        logger.info("Generating speech from response text")
        audio_file_path = synthesize_speech(response_text)
        
        # Send audio response
        with open(audio_file_path, 'rb') as audio:
            bot.send_voice(call.message.chat.id, audio)
            
        # Clean up temporary file
        os.unlink(audio_file_path)
        
    except Exception as e:
        logger.error(f"Error handling discussion button: {str(e)}", exc_info=True)
        bot.send_message(call.message.chat.id, f"I apologize, but I've encountered an obstacle in our dialogue: {str(e)}")

@bot.message_handler(commands=['clear'])
def clear_command(message):
    """
    Handle the /clear command - clears conversation history
    """
    conversation_manager.clear_history(message.from_user.id)
    bot.reply_to(message, "Our conversation history has been cleared. Let us begin anew on our intellectual journey.")

@bot.message_handler(content_types=['text'])
def handle_text(message):
    """
    Handle incoming text messages
    """
    try:
        # Check if user wants to clear history (keeping this for backward compatibility)
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
