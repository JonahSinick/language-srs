#!/usr/bin/env python3
"""
Generate translations and breakdowns for each segment using Claude API.
"""

import json
import anthropic
import time

INPUT_FILE = "segments.json"
OUTPUT_FILE = "data.json"

SYSTEM_PROMPT = """You are a Japanese language tutor. For the given Japanese text, provide:
1. A natural English translation
2. Key vocabulary (word - reading - meaning)
3. Brief grammar notes if there's anything notable

Format your response as JSON:
{
  "translation": "English translation here",
  "vocabulary": [
    {"word": "日本語", "reading": "にほんご", "meaning": "Japanese language"}
  ],
  "grammar": "Brief grammar notes or null if straightforward"
}

Keep responses concise. Only include 2-4 most important vocabulary items."""

def generate_breakdown(client, text):
    """Generate breakdown for a single segment."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"Analyze this Japanese dialogue:\n\n{text}"
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
    # Load segments
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    print(f"Loaded {len(segments)} segments")

    # Initialize client
    client = anthropic.Anthropic()

    # Process each segment
    for i, segment in enumerate(segments):
        print(f"Processing {i+1}/{len(segments)}: {segment['text'][:40]}...")

        breakdown = generate_breakdown(client, segment['text'])
        segment['translation'] = breakdown.get('translation', '')
        segment['vocabulary'] = breakdown.get('vocabulary', [])
        segment['grammar'] = breakdown.get('grammar')

        # Small delay to avoid rate limiting
        if i < len(segments) - 1:
            time.sleep(0.5)

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
