import os
from telegram.ext import Application, MessageHandler, filters, CommandHandler
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
    welcome_message = (
        "Hello! I can transcribe audio files for you.\n\n"
        "Supported formats:\n"
        "- Voice messages\n"
        "- Audio files (.mp3, .wav, .ogg, .m4a, .aac, etc.)\n\n"
        "Just send me any audio file and I'll transcribe it for you!"
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

async def send_transcript(update, transcript):
    """
    Send transcript either as a message or file depending on length.
    Max Telegram message length is 4096 characters.
    """
    # Handle both direct messages and callback queries
    message = update.message if update.message else update.callback_query.message
    
    # If transcript is short enough, send as regular message
    if len(transcript) <= 4096:
        # Escape special characters for MarkdownV2
        escaped_transcript = transcript.replace('.', '\\.').replace('-', '\\-').replace('!', '\\!').replace('(', '\\(').replace(')', '\\)')
        
        await message.reply_text(
            escaped_transcript,
            parse_mode='MarkdownV2'
        )
        return
        
    # Otherwise, send as file
    import tempfile
    
    # Get original filename from stored user data
    if hasattr(message, 'voice'):
        original_filename = f"voice_message_{message.date.strftime('%Y%m%d_%H%M%S')}"
    else:
        original_filename = os.path.splitext(message.audio.file_name)[0] if hasattr(message, 'audio') else "transcript"
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
        temp_file.write(transcript)
        temp_file_path = temp_file.name
    
    try:
        await message.reply_document(
            document=open(temp_file_path, 'rb'),
            filename=f"{original_filename}.md",
            caption="Here's your transcript as a markdown file."
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
            
            # Send transcript using the new function
            await send_transcript(update, transcript)
            
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
        
        # Run the bot
        logger.info("Bot is running...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except KeyboardInterrupt:
        logger.info("\nBot stopped gracefully")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

if __name__ == '__main__':
    main()
