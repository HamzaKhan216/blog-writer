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

REQUEST_TIMEOUT = 45

# --- MODEL CONFIGURATION ---
OUTLINE_MODEL = "gemini-2.5-flash"     # Smart thinking model for planning
WRITING_MODEL = "gemini-3.5-flash"     # Frontier writing model

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

def call_llm(system_prompt, user_prompt, model="gemini-3.1-flash-lite", json_mode=True, max_attempts=4, initial_backoff=1):
    """Attempts Gemini first with retries, falls back to Groq if all retries are exhausted."""
    # --- Gemini ---
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    gemini_payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {}
    }
    if json_mode:
        gemini_payload["generationConfig"]["responseMimeType"] = "application/json"

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Gemini ({model}) attempt {attempt}/{max_attempts}...")
            res = requests.post(gemini_url, json=gemini_payload, headers={"Content-Type": "application/json"}, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            response_text = res.json()['candidates'][0]['content']['parts'][0]['text']
            if json_mode:
                return clean_json_response(response_text)
            return response_text.strip()
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
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    if json_mode:
        groq_payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Groq attempt {attempt}/{max_attempts}...")
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=groq_payload, headers=groq_headers, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            response_text = res.json()['choices'][0]['message']['content']
            if json_mode:
                return clean_json_response(response_text)
            return response_text.strip()
        except Exception as exc:
            if attempt == max_attempts:
                print(f"Groq failed after {max_attempts} attempts.")
                raise
            backoff = initial_backoff * (2 ** (attempt - 1))
            print(f"Groq attempt {attempt} failed: {exc}. Retrying in {backoff} seconds...")
            time.sleep(backoff)

def generate_outline(trend, strategy, facts, image):
    """Uses the outline model to create a structured blog outline from all research data."""
    system_outline = (
        "You are a senior content strategist. You plan long-form blog posts that rank on Google and genuinely help readers. "
        "You think in terms of reader intent, search intent, and topical depth. Never produce thin content."
    )
    user_outline = f"""
You are planning a 2500-word blog post.

TREND: {trend['title']}
SNIPPET: {trend['snippet']}
CONTENT MODE: {trend['mode']}
TARGET KEYWORD: {strategy['target_keyword']}
ANGLE: {strategy['angle']}
SEO KEYWORDS: {strategy.get('seo_keywords', '')}
RESEARCH FACTS: {facts['summary']}
WIKIPEDIA SOURCES: {facts['links']}

TASK: Create a detailed blog outline for a 2500-word, SEO-optimized, genuinely helpful article.

REQUIREMENTS:
1. The outline must have 6-8 sections (H2s). The introduction and conclusion are separate from these.
2. Each section must have a clear H2 title, 3-5 key points to cover, and a target word count.
3. Target word counts per section should total ~2100 words (the intro and conclusion add ~400).
4. The flow must be logical: start with the most essential info, build depth, then wrap up with actionable takeaways.
5. Think about what questions the reader will have AFTER each section and answer them in the next.
6. Weave the research facts naturally into the outline where they belong.

Return ONLY valid JSON:
{{
  "title": "Compelling, keyword-rich blog title (60 chars max)",
  "meta_description": "SEO meta description (155 chars max)",
  "sections": [
    {{"h2": "Section Title", "key_points": ["point1", "point2", "point3"], "target_words": 350}}
  ]
}}
"""
    return call_llm(system_outline, user_outline, model=OUTLINE_MODEL, json_mode=True)

def write_blog_part(all_data, sections, part_number, total_parts, previous_text=""):
    """Writes one part of the blog. Part 2 receives Part 1's text to maintain flow."""
    is_first = part_number == 1
    section_list = "\n".join(
        f"- H2: {s['h2']} (cover these points: {', '.join(s['key_points'])} | ~{s['target_words']} words)"
        for s in sections
    )

    context_block = ""
    if not is_first and previous_text:
        context_block = f"""
--- TEXT ALREADY WRITTEN (Part {part_number - 1}) ---
You MUST continue naturally from where this text left off. Match its tone, sentence rhythm, and paragraph style.
Do NOT repeat any information already covered. Pick up seamlessly.

{previous_text}
--- END OF PREVIOUS TEXT ---
"""

    part_label = "the introduction + first half of the article" if is_first else "the second half of the article + conclusion"
    closing_instruction = "" if is_first else """
After the final section, write a natural conclusion paragraph that ties everything together.
Then add a "Sources" section with the Wikipedia links provided below.
"""

    system_writer = (
        "You are Maya, a former investigative journalist who now writes lifestyle guides. "
        "Your voice is conversational, slightly witty, genuinely helpful. "
        "You write like a real person who has deep expertise, not like an AI summarizing a topic. "
        "Position yourself as an authority in your niche."
    )
    user_writer = f"""
You are writing {part_label} of a {all_data['outline_word_count']}-word blog post.

TITLE: {all_data['title']}
TARGET KEYWORD: {all_data['target_keyword']}
ANGLE: {all_data['angle']}
SEO KEYWORDS: {all_data['seo_keywords']}
HERO IMAGE URL: {all_data['image_url']}
HERO IMAGE CREDIT: {all_data['image_credit']}
WIKIPEDIA SOURCES: {all_data['wikipedia_links']}
{context_block}
SECTIONS TO WRITE NOW:
{section_list}

HTML FORMAT RULES:
- Use <h2> for section headings, <p> for paragraphs, <ul>/<li> for lists.
- If this is Part 1, start with the hero image at the very top:
  <img src="{all_data['image_url']}" alt="{all_data['target_keyword']}"/>
  <p>Photo by {all_data['image_credit']} on Unsplash</p>
- Wrap the whole output in a single <div> tag.
{closing_instruction}
NON-NEGOTIABLE WRITING RULES:
1. Target ~{all_data['target_words_per_part']} words for this part.
2. Never use em-dashes (the -- character). They scream AI.
3. Never use transitional cliches: "Moreover", "In today's world", "It is important to note", "Furthermore", "Additionally".
4. Vary sentence length aggressively. Mix 8-word sentences with 30-word sentences.
5. Write like you're talking to a smart friend over coffee, not lecturing from a podium.
6. Every paragraph must earn its place. If a paragraph doesn't teach, entertain, or move the reader forward, delete it.
7. Use concrete examples and specific details, not vague generalizations.

Return ONLY valid JSON:
{{"content": "...the full HTML content for this part..."}}
"""
    return call_llm(system_writer, user_writer, model=WRITING_MODEL, json_mode=True)

def combine_blog_parts(part1_content, part2_content):
    """Merges two HTML parts into one cohesive article."""
    return f'{part1_content}\n{part2_content}'

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

    # 2. Strategy Phase (Focus on Topical Authority and Long-Tail Keywords)
    system_prompt_strategy = "You are Maya, a 32-year-old former journalist turned lifestyle blogger. You write like you're having coffee with a friend."
    user_prompt_strategy = f"""
    Trend: {trend['title']}
    Snippet: {trend['snippet']}
    TASK: Write a {'TIMELESS evergreen guide' if trend['mode'] == 'evergreen' else 'TIMELY news-angle post'}.

    CRITICAL SEO REQUIREMENT: As a brand new blog with zero authority, you CANNOT target broad keywords.
    You must extract a highly specific, LONG-TAIL KEYWORD related to this trend. Ensure the topic stays within a tight sub-niche to build TOPICAL AUTHORITY. Focus on answering a specific problem for the reader.

    Return ONLY valid JSON: {{"target_keyword": "long-tail keyword...", "search_terms": ["noun1", "noun2"], "image_query": "visual term", "seo_keywords": "5 long tail keywords", "angle": "..."}}
    """
    strategy = call_llm(system_prompt_strategy, user_prompt_strategy)

    # Extract search_terms (supports both array and legacy single-term format)
    search_terms = strategy.get('search_terms', [strategy.get('search_term', strategy['target_keyword'])])
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    # 3. Research Phase
    facts = get_wikipedia_facts(search_terms)
    image = get_unsplash_image(strategy['image_query'], fallback_query=strategy['target_keyword'])

    # 4. Outline Phase (Smart model plans the structure)
    print("--- OUTLINE PHASE ---")
    outline = generate_outline(trend, strategy, facts, image)
    print(f"Outline created: {outline['title']} ({len(outline['sections'])} sections)")

    # Split outline sections into two balanced halves
    all_sections = outline['sections']
    mid = (len(all_sections) + 1) // 2  # round up so part 1 gets the extra section if odd
    part1_sections = all_sections[:mid]
    part2_sections = all_sections[mid:]

    part1_word_count = sum(s.get('target_words', 350) for s in part1_sections) + 200  # +200 for intro
    part2_word_count = sum(s.get('target_words', 350) for s in part2_sections) + 200  # +200 for conclusion/sources
    total_word_count = part1_word_count + part2_word_count

    # Shared data bag for writing calls
    all_data = {
        "title": outline['title'],
        "target_keyword": strategy['target_keyword'],
        "angle": strategy['angle'],
        "seo_keywords": strategy.get('seo_keywords', ''),
        "image_url": image['url'],
        "image_credit": image['credit'],
        "wikipedia_links": facts['links'],
        "outline_word_count": total_word_count,
        "target_words_per_part": part1_word_count,
    }

    # 5. Writing Phase Part 1 (First half of sections)
    print("--- WRITING PART 1 ---")
    post1 = write_blog_part(all_data, part1_sections, part_number=1, total_parts=2)
    part1_html = post1.get('content', '')
    print(f"Part 1 written: {len(part1_html)} chars")

    # 6. Writing Phase Part 2 (Second half — receives Part 1 text for seamless flow)
    print("--- WRITING PART 2 ---")
    all_data["target_words_per_part"] = part2_word_count
    post2 = write_blog_part(all_data, part2_sections, part_number=2, total_parts=2, previous_text=part1_html)
    part2_html = post2.get('content', '')
    print(f"Part 2 written: {len(part2_html)} chars")

    # 7. Combine Parts
    final_html = combine_blog_parts(part1_html, part2_html)

    # 8. Validate and Publish
    if len(final_html) < 200:
        print(f"ERROR: Generated content too short or empty. Title: {outline.get('title', 'N/A')}")
        return

    publish_to_blogger(outline['title'], final_html)

if __name__ == "__main__":
    main()
