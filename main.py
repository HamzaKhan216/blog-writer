import os
import json
import random
import smtplib
import time
import requests
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# --- CONFIGURATION & SECRETS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UNSPLASH_CLIENT_ID = os.getenv("UNSPLASH_CLIENT_ID")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") # App Password
BLOGGER_EMAIL = os.getenv("BLOGGER_EMAIL")

REQUEST_TIMEOUT = 15

def get_random_trend():
    print("Fetching Google Trends...")
    response = requests.get("https://trends.google.com/trending/rss?geo=US", timeout=REQUEST_TIMEOUT)
    root = ET.fromstring(response.content)
    
    items = root.findall('.//item')[:5]
    if not items:
        raise Exception("No trends found in RSS feed.")
    
    chosen = random.choice(items)
    return {
        "title": chosen.find('title').text,
        "snippet": chosen.find('description').text if chosen.find('description') is not None else "",
        "mode": "evergreen" if random.random() < 0.6 else "news"
    }

def clean_json_response(text):
    """Removes markdown backticks from LLM responses."""
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return json.loads(text.strip())

def call_llm(system_prompt, user_prompt, max_attempts=4, initial_backoff=1):
    """Attempts Gemini first, falls back to Groq if all retries are exhausted."""
    # --- Gemini ---
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={GEMINI_API_KEY}"
    gemini_payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Gemini attempt {attempt}/{max_attempts}...")
            res = requests.post(gemini_url, json=gemini_payload, headers={"Content-Type": "application/json"}, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            response_text = res.json()['candidates'][0]['content']['parts'][0]['text']
            return clean_json_response(response_text)
        except Exception as exc:
            if attempt == max_attempts:
                print(f"Gemini failed after {max_attempts} attempts. Switching to Groq...")
            else:
                backoff = initial_backoff * (2 ** (attempt - 1))
                print(f"Gemini attempt {attempt} failed: {exc}. Retrying in {backoff} seconds...")
                time.sleep(backoff)

    # --- Groq Fallback ---
    groq_headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    groq_payload = {
        "model": "llama-3.3-70b-versatile",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Groq attempt {attempt}/{max_attempts}...")
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=groq_payload, headers=groq_headers, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            response_text = res.json()['choices'][0]['message']['content']
            return clean_json_response(response_text)
        except Exception as exc:
            if attempt == max_attempts:
                print(f"Groq failed after {max_attempts} attempts.")
                raise
            backoff = initial_backoff * (2 ** (attempt - 1))
            print(f"Groq attempt {attempt} failed: {exc}. Retrying in {backoff} seconds...")
            time.sleep(backoff)

def get_wikipedia_facts(search_terms):
    """Searches Wikipedia for multiple terms. Returns concise, combined facts."""
    print(f"Searching Wikipedia for: {search_terms}")
    headers = {"User-Agent": "AIBlogBot/1.0 (https://github.com/yourusername)"}
    
    all_summaries = []
    all_links = []

    for term in search_terms[:3]:  # max 3 terms to keep prompt lean
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": term,
            "srlimit": 2,  # top 2 results per term
            "srprop": "snippet"
        }
        try:
            res = requests.get("https://en.wikipedia.org/w/api.php", params=search_params, headers=headers, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            data = res.json()
            results = data.get('query', {}).get('search', [])
            if not results:
                continue

            page_ids = [str(item['pageid']) for item in results[:2]]
            info_params = {
                "action": "query",
                "format": "json",
                "pageids": "|".join(page_ids),
                "prop": "info",
                "inprop": "url"
            }
            info_res = requests.get("https://en.wikipedia.org/w/api.php", params=info_params, headers=headers, timeout=REQUEST_TIMEOUT)
            info_res.raise_for_status()
            pages = info_res.json().get('query', {}).get('pages', {})

            for item in results[:2]:
                title = item.get('title', 'Unknown')
                snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                page = pages.get(str(item['pageid']), {})
                url = page.get('fullurl', f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
                all_summaries.append(f"{title}: {snippet}")
                all_links.append(f"{title}: {url}")
        except Exception as e:
            print(f"Wikipedia search for '{term}' failed: {e}")
            continue

    if not all_summaries:
        return {"summary": "No specific facts found.", "links": "No Wikipedia links available."}

    return {
        "summary": " | ".join(all_summaries[:6]),   # cap at 6 to avoid prompt bloat
        "links": " | ".join(all_links[:6])
    }

def get_unsplash_image(query, fallback_query=None):
    print(f"Searching Unsplash for: {query}")
    params = {"query": query, "orientation": "landscape", "per_page": 1, "client_id": UNSPLASH_CLIENT_ID}
    res = requests.get("https://api.unsplash.com/search/photos", params=params, timeout=REQUEST_TIMEOUT)
    data = res.json()
    if data.get('results'):
        return {
            "url": data['results'][0]['urls']['regular'],
            "credit": data['results'][0]['user']['name']
        }
    
    # Fallback: try a broader term like the target keyword
    if fallback_query:
        print(f"Primary Unsplash query failed, trying fallback: {fallback_query}")
        params["query"] = fallback_query
        res = requests.get("https://api.unsplash.com/search/photos", params=params, timeout=REQUEST_TIMEOUT)
        data = res.json()
        if data.get('results'):
            return {
                "url": data['results'][0]['urls']['regular'],
                "credit": data['results'][0]['user']['name']
            }

    print("Unsplash returned no results, using placeholder.")
    return {"url": "https://images.unsplash.com/photo-1", "credit": "Unsplash"}

def publish_to_blogger(title, html_content):
    print("Publishing to Blogger via Email...")
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = BLOGGER_EMAIL
    msg['Subject'] = title
    msg.attach(MIMEText(html_content, 'html'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587, timeout=REQUEST_TIMEOUT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()
    print("Successfully Published!")

def main():
    # 1. Get Trend
    trend = get_random_trend()
    print(f"Selected Trend: {trend['title']} | Mode: {trend['mode']}")
    
    # 2. Strategy Phase
    system_prompt_strategy = "You are Maya, a 32-year-old former journalist turned lifestyle blogger. You write like you're having coffee with a friend."
    user_prompt_strategy = f"""
    Trend: {trend['title']}
    Snippet: {trend['snippet']}
    TASK: Write a {'TIMELESS evergreen guide' if trend['mode'] == 'evergreen' else 'TIMELY news-angle post'}.
    Return ONLY valid JSON: {{"target_keyword": "...", "search_terms": ["noun1", "noun2"], "image_query": "visual term", "seo_keywords": "5 keywords", "angle": "..."}}
    """
    strategy = call_llm(system_prompt_strategy, user_prompt_strategy)
    
    # Extract search_terms (supports both array and legacy single-term format)
    search_terms = strategy.get('search_terms', [strategy.get('search_term', strategy['target_keyword'])])
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    
    # 3. Research Phase
    facts = get_wikipedia_facts(search_terms)
    image = get_unsplash_image(strategy['image_query'], fallback_query=strategy['target_keyword'])
    
    # 4. Writing Phase
    system_prompt_writer = "You are Maya, a former investigative journalist who now writes lifestyle guides. Your voice is conversational, slightly witty, genuinely helpful. Use 'you' and 'I' freely."
    user_prompt_writer = f"""
    Write a blog post.
    TITLE: {strategy['target_keyword']}
    ANGLE: {strategy['angle']}
    SEO KEYWORDS (weave into headers and intro): {strategy.get('seo_keywords', '')}
    FACTS TO WEAVE IN: {facts['summary']}
    WIKIPEDIA LINKS: {facts['links']}

    Format as HTML with:
    <img src="{image['url']}" alt="{strategy['target_keyword']}"/>
    <p>Photo by {image['credit']} on Unsplash</p>

    If the article benefits from references, include the same Wikipedia links at the end as sources.
    Use clear, helpful examples and make the post rich, complete, and practical.
    Never use em-dashes "—" — they scream AI output. Avoid transitional clichés like 'Moreover,' 'In today's world,' or 'It is important to note.' Vary your sentence length: mix short punchy sentences with longer, descriptive ones.

    Use <h2>, <p>, <ul>. 900-1100 words. Return ONLY JSON: {{"title": "...", "content": "...HTML..."}}
    """
    post = call_llm(system_prompt_writer, user_prompt_writer)
    
    # 5. Validate, Save Draft, and Publish
    if not post.get('content') or len(post['content']) < 100:
        print(f"ERROR: Generated content too short or empty. Title: {post.get('title', 'N/A')}")
        return

    publish_to_blogger(post['title'], post['content'])

if __name__ == "__main__":
    main()
