#!/usr/bin/env python3
"""
Generate translations and breakdowns for Nier Replicant chapters using Claude API.
Usage: python generate_nier_translations.py 01_黒ノ病
"""

import json
import anthropic
import time
import sys
import os

SYSTEM_PROMPT = """You are a Japanese language tutor helping an intermediate learner. For the given Japanese dialogue, provide:
1. A natural English translation
2. Comprehensive vocabulary - include ALL words that an intermediate learner might not know, including:
   - Verbs (with dictionary form)
   - Adjectives
   - Adverbs (like いっぱい, ちゃんと, etc.)
   - Compound expressions
   - Grammatical words that affect meaning
3. Brief grammar notes if there's anything notable (contractions, casual speech patterns, etc.)

Format your response as JSON:
{
  "translation": "English translation here",
  "vocabulary": [
    {"word": "日本語", "reading": "にほんご", "meaning": "Japanese language"}
  ],
  "grammar": "Brief grammar notes or null if straightforward"
}

Be thorough with vocabulary - it's better to include too much than too little. Include common words if they have nuances an intermediate learner should understand."""

def generate_breakdown(client, text):
    """Generate breakdown for a single segment."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Analyze this Japanese dialogue from the video game Nier Replicant:\n\n{text}"
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
    if len(sys.argv) < 2:
        print("Usage: python generate_nier_translations.py <chapter_name>")
        print("Example: python generate_nier_translations.py 01_黒ノ病")
        sys.exit(1)

    chapter = sys.argv[1]
    chapter_dir = f"nier/{chapter}"
    input_file = f"{chapter_dir}/segments.json"
    output_file = f"{chapter_dir}/data.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    # Load segments
    with open(input_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    print(f"{'='*60}")
    print(f"CHAPTER: {chapter}")
    print(f"{'='*60}")
    print(f"Loaded {len(segments)} segments")

    # Check if we have existing data to resume from
    existing_data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
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
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"   [Saved checkpoint at {i+1} segments]")

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Saved {len(results)} segments to {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
