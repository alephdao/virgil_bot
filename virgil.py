import os
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes
import google.generativeai as genai
import aiohttp
from dotenv import load_dotenv
import logging
from telegram import Update
import base64
import gc
from contextlib import contextmanager
from google.generativeai.types import HarmCategory, HarmBlockThreshold



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

if not BOT_TOKEN or not GOOGLE_AI_API_KEY:
    raise ValueError("Missing required credentials in environment variables")

# Initialize Gemini
genai.configure(api_key=GOOGLE_AI_API_KEY)

# Add a context manager for model handling
@contextmanager
def model_context():
    """
    Context manager to handle model initialization and cleanup with safety settings
    """
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp', 
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                # These categories are defined in HarmCategory. The Gemini models only support HARM_CATEGORY_HARASSMENT, HARM_CATEGORY_HATE_SPEECH, HARM_CATEGORY_SEXUALLY_EXPLICIT, HARM_CATEGORY_DANGEROUS_CONTENT, 
            }
        )
        yield model
    finally:
        # Cleanup
        del model
        gc.collect()

# Update the prompt to be more specific
TRANSCRIPTION_PROMPT = """Transcribe this audio accurately in its original language.

If there are multiple speakers, identify and label them as 'Speaker 1:', 'Speaker 2:', etc.

Do not include any headers, titles, or additional text - only the transcription itself.

When transcribing, add line breaks between different paragraphs or distinct segments of speech to improve readability."""

# Supported audio MIME types
SUPPORTED_AUDIO_TYPES = {
    'audio/mpeg',        # .mp3
    'audio/wav',         # .wav
    'audio/ogg',         # .ogg
    'audio/x-m4a',       # .m4a
    'audio/mp4',         # .mp4 audio
    'audio/x-wav',       # alternative wav
    'audio/webm',        # .webm
    'audio/aac',         # .aac
    'audio/x-aac',       # alternative aac
}

# Maximum number of messages to keep in conversation history
MAX_HISTORY_LENGTH = 10

async def transcribe_audio(audio_data):
    """
    Transcribe audio data using Gemini API with proper cleanup
    """
    try:
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        content_parts = [
            {"text": TRANSCRIPTION_PROMPT},
            {
                "inline_data": {
                    "mime_type": "audio/mp4",
                    "data": audio_base64
                }
            }
        ]
        
        with model_context() as current_model:
            response = current_model.generate_content(content_parts)
            transcript = response.text
            
            # Log original transcript
            logger.info("Original transcript from Gemini:")
            logger.info("-" * 50)
            logger.info(transcript)
            logger.info("-" * 50)
            
            # Remove any variations of transcription headers
            transcript = transcript.replace("# Transcription\n\n", "")
            transcript = transcript.replace("Okay, here is the transcription:\n", "")
            transcript = transcript.replace("Here's the transcription:\n", "")
            transcript = transcript.strip()
            
            # Count actual speaker labels using a more precise pattern
            speaker_labels = set()
            for line in transcript.split('\n'):
                if line.strip().startswith(('Speaker ', '**Speaker ')):
                    for i in range(1, 10):
                        if f"Speaker {i}:" in line or f"**Speaker {i}:**" in line:
                            speaker_labels.add(i)
            
            # Log number of speakers detected
            logger.info(f"Number of unique speakers detected: {len(speaker_labels)}")
            logger.info(f"Speaker numbers found: {sorted(list(speaker_labels))}")
            
            # Only remove speaker labels if there's exactly one speaker
            if len(speaker_labels) == 1:
                transcript = transcript.replace("**Speaker 1:**", "")
                transcript = transcript.replace("Speaker 1:", "")
                transcript = transcript.strip()
            
            # Log cleaned transcript
            logger.info("Cleaned transcript:")
            logger.info("-" * 50)
            logger.info(transcript)
            logger.info("-" * 50)
            
            return transcript
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise
    finally:
        gc.collect()

async def start(update, context):
    """
    Handle the /start command.
    """
    # Initialize conversation history
    context.user_data['history'] = []
    
    welcome_message = (
        "Hello! I'm your AI assistant powered by Gemini.\n\n"
        "I can:\n"
        "- Chat with you via text messages\n"
        "- Transcribe and respond to voice messages\n"
        "- Process audio files (.mp3, .wav, .ogg, .m4a, .aac, etc.)\n\n"
        "Just send me a message or audio file and I'll respond!"
    )
    await update.message.reply_text(welcome_message)

async def download_file(file):
    """
    Download a file from Telegram servers.
    """
    file_obj = await file.get_file()
    async with aiohttp.ClientSession() as session:
        async with session.get(file_obj.file_path) as response:
            return await response.read()

async def send_response(update, response_text):
    """
    Send response either as a message or file depending on length.
    Max Telegram message length is 4096 characters.
    """
    # Handle both direct messages and callback queries
    message = update.message if update.message else update.callback_query.message
    
    # If response is short enough, send as regular message
    if len(response_text) <= 4096:
        await message.reply_text(response_text)
        return
        
    # Otherwise, send as file
    import tempfile
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write(response_text)
        temp_file_path = temp_file.name
    
    try:
        await message.reply_document(
            document=open(temp_file_path, 'rb'),
            filename="response.txt",
            caption="Here's my response as a text file (it was too long for a message)."
        )
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

async def handle_audio(update, context):
    """
    Handle incoming audio files and voice messages.
    """
    try:
        # Check if audio format is supported
        if update.message.audio and update.message.audio.mime_type not in SUPPORTED_AUDIO_TYPES:
            await update.message.reply_text(
                f"Sorry, the format {update.message.audio.mime_type} is not supported. "
                "Please send a common audio format like MP3, WAV, OGG, or M4A."
            )
            return
        
        # Get the audio file
        audio_file = update.message.voice or update.message.audio
        file_type = "voice message" if update.message.voice else f"audio file ({update.message.audio.mime_type})"
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            f"Processing your {file_type}... Please wait."
        )
        
        try:
            # Download and transcribe the audio
            logger.info("Downloading audio file")
            audio_data = await download_file(audio_file)
            logger.info(f"Downloaded audio file, size: {len(audio_data)} bytes")
            
            logger.info("Starting transcription")
            transcript = await transcribe_audio(audio_data)
            logger.info("Transcription completed")
            
            # Delete the processing message
            await processing_msg.delete()
            
            # Initialize conversation history if it doesn't exist
            if 'history' not in context.user_data:
                context.user_data['history'] = []
                
            # Add user message to history
            context.user_data['history'].append({"role": "user", "parts": [transcript]})
            
            # Trim history if it's too long
            if len(context.user_data['history']) > MAX_HISTORY_LENGTH:
                context.user_data['history'] = context.user_data['history'][-MAX_HISTORY_LENGTH:]
            
            # Get response from Gemini
            with model_context() as current_model:
                chat = current_model.start_chat(history=context.user_data['history'])
                response = chat.send_message(transcript)
                response_text = response.text
                
            # Add assistant response to history
            context.user_data['history'].append({"role": "model", "parts": [response_text]})
            
            # Send response
            await send_response(update, response_text)
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            error_message = (
                "Sorry, there was an error processing your audio file.\n"
                f"Error: {str(e)}\n\n"
                "Note: telegram bots have a 20MB file limit. telegram API allows 2GB."
            )
            await processing_msg.edit_text(error_message)
            
    except Exception as e:
        logger.error(f"Error handling audio file: {str(e)}")
        await update.message.reply_text(
            f"Sorry, there was an error processing your audio file. Error: {str(e)}"
        )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming text messages and maintain conversation with Gemini.
    """
    try:
        user_message = update.message.text
        
        # Initialize conversation history if it doesn't exist
        if 'history' not in context.user_data:
            context.user_data['history'] = []
        
        # Add user message to history
        context.user_data['history'].append({"role": "user", "parts": [user_message]})
        
        # Trim history if it's too long
        if len(context.user_data['history']) > MAX_HISTORY_LENGTH:
            context.user_data['history'] = context.user_data['history'][-MAX_HISTORY_LENGTH:]
        
        # Send typing action
        await update.message.chat.send_action(action="typing")
        
        # Get response from Gemini
        with model_context() as current_model:
            chat = current_model.start_chat(history=context.user_data['history'])
            response = chat.send_message(user_message)
            response_text = response.text
        
        # Add assistant response to history
        context.user_data['history'].append({"role": "model", "parts": [response_text]})
        
        # Send response
        await send_response(update, response_text)
        
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}", exc_info=True)
        await update.message.reply_text(
            f"Sorry, there was an error processing your message. Error: {str(e)}"
        )

def main():
    """Run the bot."""
    try:
        # Create and run the application
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO, 
            handle_audio
        ))
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            handle_text
        ))
        
        # Run the bot
        logger.info("Bot is running...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except KeyboardInterrupt:
        logger.info("\nBot stopped gracefully")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

if __name__ == '__main__':
    main()
