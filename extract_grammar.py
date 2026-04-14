#!/usr/bin/env python3
"""
Extract grammar patterns from the grammar text field and structure them
like vocabulary items for tracking.
"""

import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

def extract_grammar_patterns(grammar_text):
    """Extract grammar patterns from free-form grammar text."""
    if not grammar_text:
        return []

    patterns = []
    seen = set()

    # Pattern 1: ～X patterns (tilde patterns like ～ちゃダメ, ～ように)
    tilde_matches = re.findall(r'[～〜][^\s、。,\.]+', grammar_text)
    for match in tilde_matches:
        pattern = match.strip()
        if pattern and pattern not in seen:
            patterns.append({
                'pattern': pattern,
                'explanation': ''
            })
            seen.add(pattern)

    # Pattern 2: Patterns at start of text before " is ", " indicates ", " expresses ", " means "
    # e.g., "こんなもの expresses dismissal" -> "こんなもの"
    start_match = re.match(r'^([^\s]+(?:\s+[^\s]+)?)\s+(?:is|indicates|expresses|means|shows|used|forms?)\s', grammar_text, re.IGNORECASE)
    if start_match:
        pattern = start_match.group(1).strip()
        # Only if it contains Japanese characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', pattern):
            if pattern not in seen:
                patterns.append({
                    'pattern': pattern,
                    'explanation': ''
                })
                seen.add(pattern)

    # Pattern 3: Japanese text in quotes
    quote_matches = re.findall(r'[「『]([^」』]+)[」』]', grammar_text)
    for match in quote_matches:
        pattern = match.strip()
        if pattern and pattern not in seen and len(pattern) <= 20:
            patterns.append({
                'pattern': pattern,
                'explanation': ''
            })
            seen.add(pattern)

    # Pattern 4: Common grammar forms mentioned (verb forms, etc.)
    form_patterns = [
        (r'passive\s+form', 'passive form'),
        (r'causative\s+form', 'causative form'),
        (r'potential\s+form', 'potential form'),
        (r'volitional\s+form', 'volitional form'),
        (r'て-form|te-form|テ形', 'て-form'),
        (r'た-form|ta-form', 'た-form'),
        (r'ない-form|nai-form', 'ない-form'),
        (r'conditional', 'conditional'),
        (r'imperative', 'imperative'),
    ]
    for regex, name in form_patterns:
        if re.search(regex, grammar_text, re.IGNORECASE):
            if name not in seen:
                patterns.append({
                    'pattern': name,
                    'explanation': ''
                })
                seen.add(name)

    # Add the full explanation as the explanation for the first pattern
    if patterns and grammar_text:
        patterns[0]['explanation'] = grammar_text

    # If no patterns found, create one from the whole grammar text
    if not patterns and grammar_text:
        # Take the first part before any punctuation as a summary
        summary = re.split(r'[。\.!]', grammar_text)[0][:50]
        patterns.append({
            'pattern': summary,
            'explanation': grammar_text
        })

    return patterns

def process_game(game_dir, game_name):
    """Process all chapters in a game directory."""
    if not game_dir.exists():
        print(f"Directory not found: {game_dir}")
        return

    total_patterns = 0
    segments_with_grammar = 0

    for chapter_dir in sorted(game_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue

        data_path = chapter_dir / 'data.json'
        if not data_path.exists():
            continue

        with open(data_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        chapter_patterns = 0
        for seg in segments:
            grammar_text = seg.get('grammar', '')
            if grammar_text:
                grammar_items = extract_grammar_patterns(grammar_text)
                seg['grammar_patterns'] = grammar_items
                chapter_patterns += len(grammar_items)
                if grammar_items:
                    segments_with_grammar += 1
            else:
                seg['grammar_patterns'] = []

        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        total_patterns += chapter_patterns
        if chapter_patterns > 0:
            print(f"  {chapter_dir.name}: {chapter_patterns} grammar patterns")

    print(f"  Total: {total_patterns} patterns in {segments_with_grammar} segments")

def main():
    print("Extracting grammar patterns...\n")

    games = [
        ('Nier Replicant', BASE_DIR / 'nier'),
        ('Nier Automata', BASE_DIR / 'nier_automata'),
        ('Final Fantasy Tactics', BASE_DIR / 'fft'),
        ('Death Note', BASE_DIR / 'deathnote'),
    ]

    for name, game_dir in games:
        print(f"{name}:")
        process_game(game_dir, name)
        print()

    print("Done! Grammar patterns are now stored in each data.json.")

if __name__ == '__main__':
    main()
