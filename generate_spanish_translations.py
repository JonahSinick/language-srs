#!/usr/bin/env python3
"""
Generate translations and breakdowns for Spanish audio content using Claude API.
Adapted for Charlotte's Web (Spanish audiobook).
Usage: python generate_spanish_translations.py
"""

import json
import anthropic
import time
import sys
import os

# Configuration
CHAPTER_DIR = "charlottes_web/chapter_3"
INPUT_FILE = f"{CHAPTER_DIR}/segments.json"
OUTPUT_FILE = f"{CHAPTER_DIR}/data.json"

SYSTEM_PROMPT = """You are a Spanish language tutor helping an intermediate learner study Spanish through an audiobook of Charlotte's Web.

For the given Spanish text, provide:
1. A natural English translation that captures the meaning
2. Comprehensive vocabulary - include words that an intermediate learner might not know, including:
   - Verbs (with infinitive form)
   - Nouns with articles (el/la)
   - Adjectives
   - Adverbs
   - Useful expressions and phrases
3. Brief grammar notes if there's anything notable (verb tenses, subjunctive, interesting constructions, etc.)

Format your response as JSON:
{
  "translation": "English translation here",
  "vocabulary": [
    {"word": "el granero", "reading": "", "meaning": "barn"}
  ],
  "grammar": "Brief grammar notes or null if straightforward"
}

Important notes:
- The "reading" field should be left empty for Spanish (it's used for Japanese readings)
- Be thorough with vocabulary - it's better to include too much than too little
- Include common words if they have nuances or if they're frequently used
- For verbs, note the infinitive form and any irregular conjugations
- This is a children's story (Charlotte's Web) being narrated, so keep context in mind"""

def generate_breakdown(client, text):
    """Generate breakdown for a single segment."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Analyze this Spanish text from Charlotte's Web audiobook:\n\n{text}"
            }],
            system=SYSTEM_PROMPT
        )

        # Parse the JSON response
        content = response.content[0].text
        # Handle markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())
    except Exception as e:
        print(f"Error processing '{text[:30]}...': {e}")
        return {
            "translation": "",
            "vocabulary": [],
            "grammar": None
        }

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found")
        print("Run build_charlottes_web.py first to generate segments.")
        sys.exit(1)

    # Load segments
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    print(f"{'='*60}")
    print(f"GENERATING TRANSLATIONS FOR CHARLOTTE'S WEB CHAPTER 3")
    print(f"{'='*60}")
    print(f"Loaded {len(segments)} segments")

    # Check if we have existing data to resume from
    existing_data = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            # Index by text for resume capability
            for item in existing:
                if item.get('translation'):
                    existing_data[item['text']] = item
        print(f"Found {len(existing_data)} existing translations (will skip)")

    # Initialize client
    client = anthropic.Anthropic()

    # Process each segment
    results = []
    for i, segment in enumerate(segments):
        # Check if already translated
        if segment['text'] in existing_data:
            print(f"[{i+1:03d}/{len(segments)}] SKIP (exists): {segment['text'][:35]}...")
            results.append(existing_data[segment['text']])
            continue

        print(f"[{i+1:03d}/{len(segments)}] Processing: {segment['text'][:35]}...")

        breakdown = generate_breakdown(client, segment['text'])

        result = {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'audio_file': segment['audio_file'],
            'translation': breakdown.get('translation', ''),
            'vocabulary': breakdown.get('vocabulary', []),
            'grammar': breakdown.get('grammar')
        }
        results.append(result)

        # Save incrementally every 10 segments
        if (i + 1) % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"   [Saved checkpoint at {i+1} segments]")

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # Save final results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Saved {len(results)} segments to {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
