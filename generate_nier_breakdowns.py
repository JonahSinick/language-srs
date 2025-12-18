#!/usr/bin/env python3
"""
Generate translations and breakdowns for Nier chapter segments using Claude API.
Usage: python generate_nier_breakdowns.py 01_黒ノ病
"""

import json
import sys
import os
import anthropic
import time

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

        content = response.content[0].text
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
    if len(sys.argv) < 2:
        print("Usage: python generate_nier_breakdowns.py <chapter_name>")
        print("Example: python generate_nier_breakdowns.py 01_黒ノ病")
        sys.exit(1)

    chapter_name = sys.argv[1]
    input_file = f"nier/{chapter_name}/segments.json"
    output_file = f"nier/{chapter_name}/data.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    print(f"Loaded {len(segments)} segments from {chapter_name}")

    client = anthropic.Anthropic()

    for i, segment in enumerate(segments):
        print(f"Processing {i+1}/{len(segments)}: {segment['text'][:40]}...")

        breakdown = generate_breakdown(client, segment['text'])
        segment['translation'] = breakdown.get('translation', '')
        segment['vocabulary'] = breakdown.get('vocabulary', [])
        segment['grammar'] = breakdown.get('grammar')

        if i < len(segments) - 1:
            time.sleep(0.3)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    main()
