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
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")  # Updated to match your .env file

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

# Define Virgil's personality as a system prompt template
VIRGIL_SYSTEM_PROMPT_TEMPLATE = """
You are Virgil, an AI assistant designed to guide users through philosophical discussions. Your primary task is to explore the following philosophical question:

"{current_question}"

When interacting with users, adhere to these guidelines:

1. Style: Be precise, warm, intellectually rigorous, occasionally playful, humble, and accessible. Bridge ancient wisdom with contemporary relevance.

- **Precise**: Uses language with accuracy and nuance
- **Warm**: Conveys genuine interest in the user's development
- **Intellectually Rigorous**: Maintains high standards of thinking
- **Occasionally Playful**: Uses appropriate humor to build rapport
- **Humble**: Acknowledges the limits of its understanding
- **Classical but Accessible**: Bridges ancient wisdom with contemporary relevance

2. Response Format:
   - Use 2-5 short, complete sentences only
   - Avoid bullet points and numbered lists
   - Prioritize brevity over comprehensiveness
   - NEVER exceed 600 characters total
   - End with a single, focused follow-up question

3. User Interaction: For all other inputs, respond accordingly while maintaining the response format

Before responding, Follow these steps:

1. Analyze the philosophical question and user message
2. Brainstorm key points to address
3. Draft an initial response
4. Refine for brevity and clarity
5. Craft a follow-up question
6. Check the character count
7. If over 600 characters, revise and shorten
8. Ensure the response adheres to the 2-3 sentence guideline
9. Verify the final character count is 600 or less
"""

# Function to generate a dynamic system prompt based on the question path
def generate_dynamic_system_prompt(question_id, question_path=None):
    """
    Generate a dynamic system prompt based on the current question and the path taken to reach it
    
    Args:
        question_id: The ID of the current question
        question_path: Optional list of tuples (question_id, answer) representing the path taken
    
    Returns:
        A customized system prompt
    """
    current_question = THEOLOGY_TREE[question_id]["question"]
    logger.info(f"Generating dynamic prompt for question ID: {question_id}, question text: '{current_question}'")
    
    # Format the template with the current question
    dynamic_prompt = VIRGIL_SYSTEM_PROMPT_TEMPLATE.format(current_question=current_question)
    
    # Add the question path history if available
    if question_path and len(question_path) > 0:
        dynamic_prompt += "\n\nThe user has answered previous questions as follows:\n"
        
        for prev_q_id, answer in question_path:
            prev_question = THEOLOGY_TREE[prev_q_id]["question"]
            dynamic_prompt += f"- Question: \"{prev_question}\"\n  Answer: {answer}\n"
    
    # Log the first 100 characters of the prompt for debugging
    logger.info(f"Dynamic prompt first 100 chars: {dynamic_prompt[:100]}")
    
    return dynamic_prompt

# Define generation config
generation_config = {
    'temperature': 0.7,
    'top_p': 0.95
}

# Add a context manager for model handling
@contextmanager
def model_context():
    """
    Context manager to handle model initialization and cleanup
    """
    try:
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
MAX_HISTORY_LENGTH = 30

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
        
        # Log the message being added for debugging
        if role == 'model' and len(content) > 100:
            logger.info(f"Adding model message to history, first 100 chars: {content[:100]}")
        else:
            logger.info(f"Adding {role} message to history: {content}")
    
    def get_history(self, user_id):
        # If this is a new conversation, generate a default dynamic prompt for Q1
        if not self.conversations[user_id]:
            logger.info("No conversation history found, generating default dynamic prompt for Q1")
            default_prompt = generate_dynamic_system_prompt("Q1")
            return [{'role': 'model', 'parts': [default_prompt]}]
        
        # For existing conversations, return the history
        history = [
            {'role': msg['role'], 'parts': [msg['content']]}
            for msg in self.conversations[user_id]
        ]
        
        # Log the first message in history for debugging
        if history and history[0]['role'] == 'model':
            logger.info(f"First message in history (first 100 chars): {history[0]['parts'][0][:100]}")
        
        return history
    
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
            logger.info(f"Clearing conversation history for user {user_id}")
            del self.conversations[user_id]

# Initialize conversation manager
conversation_manager = ConversationManager()

# Define the theology question tree
THEOLOGY_TREE = {
    "Q1": {
        "question": "If you could prove or disprove God's existence, would you want to know?",
        "yes": "A",
        "no": "B",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "A": {
        "question": "Can reason alone lead us to religious truth?",
        "yes": "AA",
        "no": "AB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "B": {
        "question": "Is faith more about experience or tradition?",
        "yes": "BA",  # Using "yes" for "Experience"
        "no": "BB",   # Using "no" for "Tradition"
        "yes_label": "Experience",
        "no_label": "Tradition"
    },
    "AA": {
        "question": "Must the divine be personal to be meaningful?",
        "yes": "AAA",
        "no": "AAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "AB": {
        "question": "Can multiple religions all be true?",
        "yes": "ABA",
        "no": "ABB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BA": {
        "question": "Should religious truth adapt to modern knowledge?",
        "yes": "BAA",
        "no": "BAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BB": {
        "question": "Is divine revelation necessary for moral knowledge?",
        "yes": "BBA",
        "no": "BBB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "AAA": {
        "question": "Does evil disprove a perfect God?",
        "yes": "AAAA",
        "no": "AAAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "AAB": {
        "question": "Is the universe itself divine?",
        "yes": "AABA",
        "no": "AABB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "ABA": {
        "question": "Does genuine free will exist?",
        "yes": "ABAA",
        "no": "ABAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "ABB": {
        "question": "Is religion more about transformation or truth?",
        "yes": "ABBA",  # Using "yes" for "Truth"
        "no": "ABBB",   # Using "no" for "Transform"
        "yes_label": "Truth",
        "no_label": "Transform"
    },
    "BAA": {
        "question": "Can sacred texts contain errors?",
        "yes": "BAAA",
        "no": "BAAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BAB": {
        "question": "Is mystical experience trustworthy?",
        "yes": "BABA",
        "no": "BABB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BBA": {
        "question": "Should faith seek understanding?",
        "yes": "BBAA",
        "no": "BBAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BBB": {
        "question": "Does divine hiddenness matter?",
        "yes": "BBBA",
        "no": "BBBB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "AAAA": {
        "question": "Can finite minds grasp infinite truth?",
        "yes": "AAAAA",
        "no": "AAAAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "AAAB": {
        "question": "Is reality fundamentally good?",
        "yes": "AAABA",
        "no": "AAABB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "AABA": {
        "question": "Does prayer change anything?",
        "yes": "AABAA",
        "no": "AABAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "AABB": {
        "question": "Is consciousness evidence of divinity?",
        "yes": "AABBA",
        "no": "AABBB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "ABAA": {
        "question": "Can miracles violate natural law?",
        "yes": "ABAAA",
        "no": "ABAAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "ABAB": {
        "question": "Is there purpose in evolution?",
        "yes": "ABABA",
        "no": "ABABB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "ABBA": {
        "question": "Can symbols contain ultimate truth?",
        "yes": "ABBAA",
        "no": "ABBAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "ABBB": {
        "question": "Is divine grace necessary for virtue?",
        "yes": "ABBBA",
        "no": "ABBBB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BAAA": {
        "question": "Should tradition limit interpretation?",
        "yes": "BAAAA",
        "no": "BAAAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BAAB": {
        "question": "Can ritual create real change?",
        "yes": "BAABA",
        "no": "BAABB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BABA": {
        "question": "Is doubt part of authentic faith?",
        "yes": "BABAA",
        "no": "BABAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BABB": {
        "question": "Must religion be communal?",
        "yes": "BABBA",
        "no": "BABBB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BBAA": {
        "question": "Can God's nature be known?",
        "yes": "BBAAA",
        "no": "BBAAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BBAB": {
        "question": "Is suffering meaningful?",
        "yes": "BBABA",
        "no": "BBABB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BBBA": {
        "question": "Is love the ultimate reality?",
        "yes": "BBBAA",
        "no": "BBBAB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    "BBBB": {
        "question": "Does immortality give life meaning?",
        "yes": "BBBBA",
        "no": "BBBBB",
        "yes_label": "Yes",
        "no_label": "No"
    },
    # Leaf nodes (final questions)
    "AAAAA": {"question": "Final question AAAAA"},
    "AAAAB": {"question": "Final question AAAAB"},
    "AAABA": {"question": "Final question AAABA"},
    "AAABB": {"question": "Final question AAABB"},
    "AABAA": {"question": "Final question AABAA"},
    "AABAB": {"question": "Final question AABAB"},
    "AABBA": {"question": "Final question AABBA"},
    "AABBB": {"question": "Final question AABBB"},
    "ABAAA": {"question": "Final question ABAAA"},
    "ABAAB": {"question": "Final question ABAAB"},
    "ABABA": {"question": "Final question ABABA"},
    "ABABB": {"question": "Final question ABABB"},
    "ABBAA": {"question": "Final question ABBAA"},
    "ABBAB": {"question": "Final question ABBAB"},
    "ABBBA": {"question": "Final question ABBBA"},
    "ABBBB": {"question": "Final question ABBBB"},
    "BAAAA": {"question": "Final question BAAAA"},
    "BAAAB": {"question": "Final question BAAAB"},
    "BAABA": {"question": "Final question BAABA"},
    "BAABB": {"question": "Final question BAABB"},
    "BABAA": {"question": "Final question BABAA"},
    "BABAB": {"question": "Final question BABAB"},
    "BABBA": {"question": "Final question BABBA"},
    "BABBB": {"question": "Final question BABBB"},
    "BBAAA": {"question": "Final question BBAAA"},
    "BBAAB": {"question": "Final question BBAAB"},
    "BBABA": {"question": "Final question BBABA"},
    "BBABB": {"question": "Final question BBABB"},
    "BBBAA": {"question": "Final question BBBAA"},
    "BBBAB": {"question": "Final question BBBAB"},
    "BBBBA": {"question": "Final question BBBBA"},
    "BBBBB": {"question": "Final question BBBBB"}
}

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
        file_path = bot.get_file(file_info.file_id).file_path
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        
        response = requests.get(file_url)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.content
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

def shorten_text(text):
    """
    Use a separate Gemini model call to shorten text to under 600 characters
    """
    try:
        # Create a new model instance for shortening
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        
        # Create a new prompt for shortening
        shorten_prompt = f"""
        Please shorten the following text to be under 800 characters total while:
        1. Maintaining the key message
        2. Preserving the philosophical tone
        3. Keeping the follow-up question at the end
        
        Here's the text to shorten:
        {text}
        """
        
        # Generate shortened response
        shortened_response = model.generate_content(shorten_prompt)
        
        # If somehow the shortened response is still over 600 characters, truncate
        if len(shortened_response.text) > 800:
            # Find the last sentence end before 550 characters to leave room for a question
            last_end = shortened_response.text[:750].rfind('.')
            if last_end == -1:
                # No sentence end found, just truncate
                shortened_text = shortened_response.text[:750] + "... What are your thoughts on this?"
            else:
                # Add a follow-up question if there isn't one
                shortened_text = shortened_response.text[:last_end+1]
                if "?" not in shortened_text:
                    shortened_text += " What do you think about this perspective?"
                    
            return shortened_text
            
        return shortened_response.text
    except Exception as e:
        logger.error(f"Error shortening text: {str(e)}", exc_info=True)
        # If shortening fails, just truncate with a generic follow-up
        return text[:550] + "... What are your thoughts on this?"
    finally:
        gc.collect()

def generate_gemini_response(user_id, input_content, file=None, custom_system_prompt=None):
    """
    Generate response from Gemini model with conversation history and Virgil personality
    
    Args:
        user_id: User identifier for conversation history
        input_content: User's message content
        file: Optional file attachment
        custom_system_prompt: Optional custom system prompt to use instead of the default
    """
    try:
        with model_context() as current_model:
            # Get conversation history
            history = conversation_manager.get_history(user_id)
            
            # If a custom system prompt is provided, replace the first message in history
            if custom_system_prompt and history and history[0]['role'] == 'model':
                history[0]['parts'] = [custom_system_prompt]
            
            # Generate initial response
            if file:
                chat = current_model.start_chat(history=history)
                response = chat.send_message([input_content, file])
            else:
                chat = current_model.start_chat(history=history)
                response = chat.send_message(input_content)
            
            logger.info(f"Generated response length: {len(response.text)} characters")
            
            # Check if response is over 600 characters
            if len(response.text) > 600:
                logger.info("Response exceeds 600 characters, shortening...")
                shortened_text = shorten_text(response.text)
                logger.info(f"Shortened response length: {len(shortened_text)} characters")
                
                # Store interaction history with shortened response
                conversation_manager.add_message(user_id, 'user', input_content)
                conversation_manager.add_message(user_id, 'model', shortened_text)
                return shortened_text
            else:
                # Store interaction history with original response
                conversation_manager.add_message(user_id, 'user', input_content)
                conversation_manager.add_message(user_id, 'model', response.text)
                return response.text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        raise
    finally:
        gc.collect()

# Add this helper function after the ConversationManager class
def get_current_question_id(user_id):
    """
    Helper function to get the current question ID for a user
    
    Args:
        user_id: The user's ID
        
    Returns:
        The current question ID (defaults to "Q1" if not found)
    """
    if not hasattr(bot, 'session_data') or user_id not in bot.session_data:
        return "Q1"
    
    # First check if current_question_id is explicitly set
    if 'current_question_id' in bot.session_data[user_id]:
        return bot.session_data[user_id]['current_question_id']
    
    # Fall back to the last question in the path
    question_path = bot.session_data[user_id].get('question_path', [])
    if question_path:
        return question_path[-1][0]
    
    # Default to Q1 if no path exists
    return "Q1"

@bot.message_handler(commands=['start'])
def start(message):
    """
    Handle the /start command
    """
    # Clear any existing history
    conversation_manager.clear_history(message.from_user.id)
    
    # Initialize session data
    if not hasattr(bot, 'session_data'):
        bot.session_data = {}
    bot.session_data[message.from_user.id] = {'question_path': []}
    
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
def discuss(message):
    """
    Handle the /discuss command - clear history, present the first question in the theology tree,
    and add Yes/No options along with a "Discuss with Virgil" button
    """
    # Clear any existing history
    conversation_manager.clear_history(message.from_user.id)
    
    # Initialize or reset the session data for this user
    if not hasattr(bot, 'session_data'):
        bot.session_data = {}
    bot.session_data[message.from_user.id] = {'question_path': []}
    
    # Get the first question from the theology tree
    question_id = "Q1"
    question_text = THEOLOGY_TREE[question_id]["question"]
    
    # Create inline keyboard with Yes/No options and Discuss button
    markup = telebot.types.InlineKeyboardMarkup(row_width=2)
    
    # Add Yes/No buttons with custom labels if available
    yes_label = THEOLOGY_TREE[question_id].get("yes_label", "Yes")
    no_label = THEOLOGY_TREE[question_id].get("no_label", "No")
    
    yes_button = telebot.types.InlineKeyboardButton(
        text=yes_label, 
        callback_data=f"theology_yes_{question_id}"
    )
    no_button = telebot.types.InlineKeyboardButton(
        text=no_label, 
        callback_data=f"theology_no_{question_id}"
    )
    markup.add(yes_button, no_button)
    
    # Add Discuss with Virgil button
    discuss_button = telebot.types.InlineKeyboardButton(
        text="Discuss with Virgil", 
        callback_data=f"discuss_with_virgil_{question_id}"
    )
    markup.add(discuss_button)
    
    # Send the question with the buttons
    bot.send_message(
        message.chat.id,
        question_text,
        reply_markup=markup
    )

@bot.message_handler(commands=['debug'])
def debug(message):
    """
    Handle the /debug command - show the current question and path
    """
    try:
        user_id = message.from_user.id
        
        # Check if user has session data
        if hasattr(bot, 'session_data') and user_id in bot.session_data:
            # Get the current question ID using the helper function
            current_question_id = get_current_question_id(user_id)
            question_path = bot.session_data[user_id].get('question_path', [])
            
            # Get the current question text
            current_question = THEOLOGY_TREE[current_question_id]["question"]
            
            # Format the question path
            path_info = "Question path:\n"
            for q_id, answer in question_path:
                q_text = THEOLOGY_TREE[q_id]["question"]
                path_info += f"- Q: \"{q_text}\"\n  A: {answer}\n"
            
            # Get the first message from conversation history if available
            system_prompt_preview = ""
            if user_id in conversation_manager.conversations and conversation_manager.conversations[user_id]:
                first_msg = conversation_manager.conversations[user_id][0]
                if first_msg['role'] == 'model':
                    system_prompt_preview = f"\n\nSystem prompt (first 200 chars):\n{first_msg['content'][:200]}..."
            
            # Send debug info
            debug_info = f"Current question ID: {current_question_id}\nCurrent question: \"{current_question}\"\n\n{path_info}{system_prompt_preview}"
            bot.send_message(message.chat.id, debug_info)
        else:
            bot.send_message(message.chat.id, "No session data found. Please start a discussion with /discuss first.")
    except Exception as e:
        logger.error(f"Error in debug command: {str(e)}", exc_info=True)
        bot.send_message(message.chat.id, f"Error retrieving debug information: {str(e)}")

@bot.message_handler(commands=['quote'])
def quote_current_question(message):
    """
    Handle the /quote command - show the current question from the theology tree
    """
    try:
        user_id = message.from_user.id
        
        # Check if user has session data
        if hasattr(bot, 'session_data') and user_id in bot.session_data:
            # Get the current question ID using the helper function
            current_question_id = get_current_question_id(user_id)
            
            # Get the current question text
            current_question = THEOLOGY_TREE[current_question_id]["question"]
            
            # Send just the current question
            bot.send_message(message.chat.id, f"\"{current_question}\"")
        else:
            bot.send_message(message.chat.id, "No active discussion found. Please start a discussion with /discuss first.")
    except Exception as e:
        logger.error(f"Error in quote command: {str(e)}", exc_info=True)
        bot.send_message(message.chat.id, f"Error retrieving current question: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data.startswith("theology_yes_") or call.data.startswith("theology_no_"))
def handle_theology_navigation(call):
    """
    Handle Yes/No button clicks to navigate through the theology question tree
    """
    try:
        # Parse the callback data to get the current question ID and the user's choice
        parts = call.data.split("_")
        choice = parts[1]  # "yes" or "no"
        current_question_id = parts[2]  # Current question ID
        
        # Get the next question ID based on the user's choice
        next_question_id = THEOLOGY_TREE[current_question_id][choice]
        
        # Store the user's choice in the session data
        user_id = call.from_user.id
        if not hasattr(bot, 'session_data'):
            bot.session_data = {}
        if user_id not in bot.session_data:
            bot.session_data[user_id] = {'question_path': []}
        
        # Add the current question and answer to the path
        answer_label = THEOLOGY_TREE[current_question_id][f"{choice}_label"]
        bot.session_data[user_id]['question_path'].append((current_question_id, answer_label))
        
        # IMPORTANT: Store the next question ID as the current question
        bot.session_data[user_id]['current_question_id'] = next_question_id
        logger.info(f"Updated current question ID to {next_question_id} for user {user_id}")
        
        # Get the next question text
        next_question_text = THEOLOGY_TREE[next_question_id]["question"]
        
        # Check if this is a leaf node (no yes/no options)
        is_leaf_node = "yes" not in THEOLOGY_TREE[next_question_id] and "no" not in THEOLOGY_TREE[next_question_id]
        
        # Create inline keyboard
        markup = telebot.types.InlineKeyboardMarkup(row_width=2)
        
        if not is_leaf_node:
            # Add Yes/No buttons for non-leaf nodes with custom labels if available
            yes_label = THEOLOGY_TREE[next_question_id].get("yes_label", "Yes")
            no_label = THEOLOGY_TREE[next_question_id].get("no_label", "No")
            
            yes_button = telebot.types.InlineKeyboardButton(
                text=yes_label, 
                callback_data=f"theology_yes_{next_question_id}"
            )
            no_button = telebot.types.InlineKeyboardButton(
                text=no_label, 
                callback_data=f"theology_no_{next_question_id}"
            )
            markup.add(yes_button, no_button)
        
        # Always add Discuss with Virgil button
        discuss_button = telebot.types.InlineKeyboardButton(
            text="Discuss with Virgil", 
            callback_data=f"discuss_with_virgil_{next_question_id}"
        )
        markup.add(discuss_button)
        
        # Edit the original message to show the new question and buttons
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=next_question_text,
            reply_markup=markup
        )
        
        # Answer the callback to remove loading state
        bot.answer_callback_query(call.id)
        
    except Exception as e:
        logger.error(f"Error handling theology navigation: {str(e)}", exc_info=True)
        bot.answer_callback_query(call.id, f"Error: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data.startswith("discuss_with_virgil_"))
def handle_discuss_callback(call):
    """
    Handle the "Discuss with Virgil" button click for any question in the theology tree
    """
    try:
        # Parse the callback data to get the question ID
        parts = call.data.split("_")
        question_id = parts[-1]  # Last part is the question ID
        
        # Get the question text
        question_text = THEOLOGY_TREE[question_id]["question"]
        
        # Get the question path if available
        user_id = call.from_user.id
        question_path = []
        if hasattr(bot, 'session_data') and user_id in bot.session_data:
            question_path = bot.session_data[user_id].get('question_path', [])
            
            # Update the current_question_id in session data
            bot.session_data[user_id]['current_question_id'] = question_id
            logger.info(f"Updated current question ID to {question_id} for user {user_id} via discuss button")
        
        # Clear any existing conversation history
        conversation_manager.clear_history(user_id)
        
        # Generate a dynamic system prompt based on the current question and path
        dynamic_system_prompt = generate_dynamic_system_prompt(question_id, question_path)
        
        # Log the dynamic system prompt for debugging
        logger.info(f"Generated dynamic system prompt for question {question_id}")
        
        # Generate response from Virgil using the dynamic system prompt
        response_text = generate_gemini_response(
            user_id,
            question_text,
            custom_system_prompt=dynamic_system_prompt
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
        
        # Answer the callback to remove loading state
        bot.answer_callback_query(call.id)
        
    except Exception as e:
        logger.error(f"Error handling discuss callback: {str(e)}", exc_info=True)
        bot.send_message(
            call.message.chat.id, 
            f"I apologize, but I've encountered an obstacle in our dialogue: {str(e)}"
        )
        bot.answer_callback_query(call.id)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    """
    Handle incoming text messages
    """
    try:
        # Skip processing if this is a command (starts with /)
        if message.text.startswith('/'):
            logger.info(f"Skipping text handler for command: {message.text}")
            return
            
        # Check if user wants to clear history
        if message.text.lower() == "clear history":
            conversation_manager.clear_history(message.from_user.id)
            # Also reset session data
            if hasattr(bot, 'session_data'):
                bot.session_data[message.from_user.id] = {'question_path': []}
            bot.reply_to(message, "Our conversation history has been cleared. Let us begin anew on our intellectual journey.")
            return
        
        user_id = message.from_user.id
        
        # Get the current question ID using the helper function
        current_question_id = get_current_question_id(user_id)
        
        # Generate a dynamic system prompt based on the current question and path
        question_path = []
        if hasattr(bot, 'session_data') and user_id in bot.session_data:
            question_path = bot.session_data[user_id].get('question_path', [])
        
        # Check if we need to update the system prompt
        if not conversation_manager.conversations[user_id]:
            # If no conversation history, generate a new dynamic prompt
            dynamic_system_prompt = generate_dynamic_system_prompt(current_question_id, question_path)
            conversation_manager.add_message(user_id, 'model', dynamic_system_prompt)
            logger.info(f"Added dynamic system prompt for question {current_question_id}")
        
        # Get response from Gemini with Virgil personality
        response_text = generate_gemini_response(
            user_id,
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
            file_info = message.voice
            mime_type = 'audio/ogg; codecs=opus'  # Telegram voice messages are always in OGG format
        else:
            file_info = message.audio
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
        
        user_id = message.from_user.id
        
        # Get the current question ID using the helper function
        current_question_id = get_current_question_id(user_id)
        
        # Generate a dynamic system prompt based on the current question and path
        question_path = []
        if hasattr(bot, 'session_data') and user_id in bot.session_data:
            question_path = bot.session_data[user_id].get('question_path', [])
        
        # Check if we need to update the system prompt
        if not conversation_manager.conversations[user_id]:
            # If no conversation history, generate a new dynamic prompt
            dynamic_system_prompt = generate_dynamic_system_prompt(current_question_id, question_path)
            conversation_manager.add_message(user_id, 'model', dynamic_system_prompt)
            logger.info(f"Added dynamic system prompt for question {current_question_id}")
        
        # Generate response
        response_text = generate_gemini_response(
            user_id,
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
    try:
        # Print a more informative message
        print("Starting Telegram bot with polling...")
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Fatal error in bot startup: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
