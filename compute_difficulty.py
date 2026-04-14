#!/usr/bin/env python3
"""
Precompute difficulty scores for all segments based on Japanese word frequency.
Stores the difficulty score directly in each segment's data.json.
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load JLPT-based frequency data (better for learners)
with open(BASE_DIR / 'jlpt_frequency.json', 'r', encoding='utf-8') as f:
    jp_freq = json.load(f)

DEFAULT_RANK = 20000  # For words not in frequency list

def calculate_difficulty(vocabulary):
    """Calculate difficulty based on hardest (rarest) unknown word."""
    if not vocabulary:
        return 999999  # No vocab = sort to end

    max_rank = 0
    hardest_word = ''

    for v in vocabulary:
        word = v.get('word', '')
        rank = jp_freq.get(word, DEFAULT_RANK)
        if rank > max_rank:
            max_rank = rank
            hardest_word = word

    # Score = hardest word's rank + (vocab count * 100)
    score = max_rank + len(vocabulary) * 100

    return score, hardest_word, max_rank

def process_game(game_dir, game_name):
    """Process all chapters in a game directory."""
    if not game_dir.exists():
        print(f"Directory not found: {game_dir}")
        return

    total_segments = 0

    for chapter_dir in sorted(game_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue

        data_path = chapter_dir / 'data.json'
        if not data_path.exists():
            continue

        with open(data_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        for seg in segments:
            vocab = seg.get('vocabulary', [])
            if vocab:
                score, hardest, rank = calculate_difficulty(vocab)
                seg['difficulty'] = score
                seg['hardest_word'] = hardest
                seg['hardest_rank'] = rank
            else:
                seg['difficulty'] = 999999
                seg['hardest_word'] = ''
                seg['hardest_rank'] = 0
            total_segments += 1

        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f"  {chapter_dir.name}: {len(segments)} segments")

    print(f"  Total: {total_segments} segments processed")

def main():
    print("Computing difficulty scores...\n")

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

    print("Done! Difficulty scores are now stored in each data.json.")

if __name__ == '__main__':
    main()
