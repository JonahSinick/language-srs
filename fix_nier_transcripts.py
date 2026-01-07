#!/usr/bin/env python3
"""
Script to fix incorrect kanji in Nier Replicant transcripts based on vocabulary breakdowns.

When the automatic transcription picks the wrong homophone (same reading, different kanji),
this script detects and corrects it using the vocabulary data which has the correct kanji.
"""

import json
import glob
import pykakasi
from pathlib import Path

def get_reading(text, kakasi):
    """Get hiragana reading for Japanese text."""
    result = kakasi.convert(text)
    return ''.join([item['hira'] for item in result])

def find_and_replace_by_reading(transcript, vocab_word, vocab_reading, kakasi):
    """
    Find text in transcript that has the same reading as vocab_word but different kanji.
    Returns (modified_transcript, was_changed, old_text).
    """
    # Skip if vocab word is already in transcript
    if vocab_word in transcript:
        return transcript, False, None

    # Skip if no reading provided or reading is same as word (no kanji)
    if not vocab_reading or vocab_reading == vocab_word:
        return transcript, False, None

    # Skip vocab words that look like they have grammar markers (〜, etc)
    if '〜' in vocab_word or '～' in vocab_word:
        return transcript, False, None

    # Blocklist of problematic corrections (vocab entry may be wrong or replacement breaks text)
    blocklist = {
        '指者',  # Not a real word, vocab entry error
        '目',    # Single char replacement breaks メモ → 目モ
    }
    if vocab_word in blocklist:
        return transcript, False, None

    # Normalize the vocab reading to hiragana
    vocab_reading_hira = get_reading(vocab_reading, kakasi)

    # Try to find a segment in the transcript with matching reading
    # Only look for EXACT same length replacements (safer for kanji swaps)
    word_len = len(vocab_word)

    for i in range(len(transcript) - word_len + 1):
        segment = transcript[i:i + word_len]

        # Skip if segment is already the correct word
        if segment == vocab_word:
            continue

        # Skip segments that are clearly not kanji (pure hiragana/katakana/ascii)
        segment_reading_check = get_reading(segment, kakasi)
        if segment == segment_reading_check:
            continue

        segment_reading = get_reading(segment, kakasi)

        # Check if readings match exactly
        if segment_reading == vocab_reading_hira:
            # Extra safety: make sure we're not breaking a longer word
            # Check that the replacement doesn't create obvious grammar errors
            # by ensuring we're replacing kanji with kanji (not inserting hiragana)

            # Count kanji in both
            def count_kanji(s):
                return sum(1 for c in s if '\u4e00' <= c <= '\u9fff')

            # Only proceed if vocab word has at least as many kanji as segment
            # This prevents replacing 進 with 進む (adding hiragana)
            if count_kanji(vocab_word) < count_kanji(segment):
                continue

            # Found a match - replace it
            new_transcript = transcript[:i] + vocab_word + transcript[i + word_len:]
            return new_transcript, True, segment

    return transcript, False, None

def process_data_file(filepath, kakasi, dry_run=False):
    """Process a single data.json file and fix transcripts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changes = []

    for idx, slide in enumerate(data):
        text = slide.get('text', '')
        vocabulary = slide.get('vocabulary', []) or []

        original_text = text

        for vocab in vocabulary:
            word = vocab.get('word', '')
            reading = vocab.get('reading', '')

            if not word:
                continue

            text, was_changed, old_text = find_and_replace_by_reading(
                text, word, reading, kakasi
            )

            if was_changed:
                changes.append({
                    'file': str(filepath),
                    'slide': idx + 1,
                    'old_text': old_text,
                    'new_text': word,
                    'reading': reading,
                    'full_original': original_text,
                    'full_corrected': text
                })

        if text != original_text:
            slide['text'] = text

    if changes and not dry_run:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return changes

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix Nier transcript kanji errors')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    args = parser.parse_args()

    kakasi = pykakasi.kakasi()

    # Find all data.json files in nier folder
    data_files = glob.glob('nier/*/data.json')

    all_changes = []

    for filepath in sorted(data_files):
        print(f"Processing {filepath}...")
        changes = process_data_file(filepath, kakasi, dry_run=args.dry_run)
        all_changes.extend(changes)

    # Report changes
    print(f"\n{'=' * 60}")
    print(f"Found {len(all_changes)} corrections")
    print('=' * 60)

    for change in all_changes:
        print(f"\nFile: {change['file']}")
        print(f"Slide: {change['slide']}")
        print(f"  {change['old_text']} → {change['new_text']} (reading: {change['reading']})")
        print(f"  Original: {change['full_original']}")
        print(f"  Corrected: {change['full_corrected']}")

    if args.dry_run:
        print(f"\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")
    else:
        print(f"\n{len(all_changes)} corrections applied to data files.")

if __name__ == '__main__':
    main()
