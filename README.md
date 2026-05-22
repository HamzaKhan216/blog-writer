# AI Blog Bot

An autonomous blog post generator that scrapes Google Trends, researches topics via Wikipedia, generates content using Gemini AI (with Groq fallback), grabs images from Unsplash, and publishes to Blogger via email.

## How It Works

1. **Trend Picking** — Fetches the top 5 Google Trends (US RSS feed) and picks one at random. 60% chance of evergreen content, 40% news-angle.
2. **Strategy** — Sends the trend to Gemini to define a target keyword, search terms, image query, SEO keywords, and content angle.
3. **Research** — Searches Wikipedia for the chosen search terms and pulls real facts and source links.
    10|4. **Image** — Queries Unsplash for a relevant landscape photo.
    11|5. **Writing** — Generates a full 900–1100 word HTML blog post in Maya's voice (conversational lifestyle blogger).
    12|6. **Publish** — Emails the post to Blogger via Gmail SMTP.
    13|
    14|## Requirements

- Python 3.8+
- `requests` (see `requirements.txt`)

## Setup

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables (or edit the defaults in `main.py`):

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (fallback LLM) |
| `GEMINI_API_KEY` | Google Gemini API key (primary LLM) |
| `UNSPLASH_CLIENT_ID` | Unsplash API access key |
| `SENDER_EMAIL` | Gmail address for sending |
| `SENDER_PASSWORD` | Gmail app password |
| `BLOGGER_EMAIL` | Blogger email address (Mail-to-Blogger) |

## Usage

```bash
python main.py
```

## Configuration

All config lives at the top of `main.py`:
    47|- `REQUEST_TIMEOUT` — HTTP request timeout in seconds (default: 15)
    48|- Retry logic applies exponential backoff (up to 4 attempts) for both Gemini and Groq calls
