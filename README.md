# Virgil Bot

A philosophical Telegram bot inspired by Dante's guide through the Divine Comedy. Virgil Bot engages users in thoughtful discussions about the relationship between reason and religious truth.

## Features

- **Philosophical Guidance**: Explores the question "Can reason alone lead us to religious truth?"
- **Voice Interaction**: Supports both text and voice message interactions
- **Concise Responses**: Delivers thoughtful but concise responses (under 600 characters)
- **British Voice**: Uses Amazon Polly's "Arthur" voice for audio responses

## Technical Stack

- **AI Engine**: Google Gemini 2.0 Flash
- **Bot Framework**: Telegram Bot API (pyTelegramBotAPI)
- **Text-to-Speech**: Amazon Polly
- **Language**: Python

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following environment variables:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   GOOGLE_AI_API_KEY=your_gemini_api_key
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_DEFAULT_REGION=us-east-1
   ```
4. Run the bot:
   ```
   python virgil.py
   ```

## Usage

- Start the bot with the `/discuss` command in Telegram
- Send text or voice messages to engage in philosophical discussion
- Write /discuss to clear the history and start again
