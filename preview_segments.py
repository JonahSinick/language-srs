#!/usr/bin/env python3
"""
Preview merged segments without splitting audio.
"""

import re

TRANSCRIPT_PATH = "source_media/episode01.txt"

# Time ranges to exclude (opening song, ending song, preview)
EXCLUDE_RANGES = [
    (0, 112),      # 00:00-01:52 Opening song
    (1344, 1400),  # 22:24-23:20 Ending song + preview
]

# Merge settings
MIN_SEGMENT_DURATION = 3.0  # Merge if under this duration
MAX_GAP_TO_MERGE = 1.5      # Merge if gap to next is under this
MAX_MERGED_DURATION = 10.0  # Stop merging if total exceeds this
END_BUFFER = 0.4            # Add this many seconds to end of each clip

def time_to_seconds(time_str):
    """Convert MM:SS to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])

def seconds_to_time(seconds):
    """Convert seconds to MM:SS format."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"

def is_excluded(start_sec):
    """Check if a timestamp falls within excluded ranges."""
    for exc_start, exc_end in EXCLUDE_RANGES:
        if exc_start <= start_sec <= exc_end:
            return True
    return False

def is_english(text):
    """Check if text is primarily English."""
    # Simple heuristic: if more than half the characters are ASCII letters
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return ascii_letters / total_letters > 0.5

def parse_transcript(filepath):
    """Parse the transcript file and extract segments."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if '→' in line:
            line = line.split('→', 1)[1]

        timestamp_match = re.match(r'^(\d{2}:\d{2})-(\d{2}:\d{2})$', line)
        if timestamp_match:
            start_time = timestamp_match.group(1)
            end_time = timestamp_match.group(2)
            start_sec = time_to_seconds(start_time)
            end_sec = time_to_seconds(end_time)

            i += 1
            text_lines = []
            while i < len(lines):
                text_line = lines[i].strip()
                if '→' in text_line:
                    text_line = text_line.split('→', 1)[1]
                if re.match(r'^(\d{2}:\d{2})-(\d{2}:\d{2})$', text_line) or text_line == '':
                    break
                text_lines.append(text_line)
                i += 1

            text = ' '.join(text_lines)

            # Filter out excluded ranges and English
            if text and not is_excluded(start_sec) and not is_english(text):
                segments.append({
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'text': text
                })
        else:
            i += 1

    return segments

def merge_segments(segments):
    """Merge short/consecutive segments intelligently."""
    if not segments:
        return []

    merged = []
    current = {
        'start_sec': segments[0]['start_sec'],
        'end_sec': segments[0]['end_sec'],
        'texts': [segments[0]['text']]
    }

    for i in range(1, len(segments)):
        seg = segments[i]
        current_duration = current['end_sec'] - current['start_sec']
        gap = seg['start_sec'] - current['end_sec']

        # Decide whether to merge
        should_merge = False

        if current_duration < MIN_SEGMENT_DURATION:
            # Current segment is too short, try to merge
            if gap <= MAX_GAP_TO_MERGE:
                should_merge = True
        elif gap <= 0.5:
            # Very small gap, likely continuous speech
            potential_duration = seg['end_sec'] - current['start_sec']
            if potential_duration <= MAX_MERGED_DURATION:
                should_merge = True

        if should_merge:
            current['end_sec'] = seg['end_sec']
            current['texts'].append(seg['text'])
        else:
            # Finalize current and start new
            merged.append({
                'start_sec': current['start_sec'],
                'end_sec': current['end_sec'] + END_BUFFER,
                'text': ' / '.join(current['texts']),
                'original_count': len(current['texts'])
            })
            current = {
                'start_sec': seg['start_sec'],
                'end_sec': seg['end_sec'],
                'texts': [seg['text']]
            }

    # Don't forget the last one
    merged.append({
        'start_sec': current['start_sec'],
        'end_sec': current['end_sec'] + END_BUFFER,
        'text': ' / '.join(current['texts']),
        'original_count': len(current['texts'])
    })

    return merged

def main():
    print("Parsing transcript...")
    segments = parse_transcript(TRANSCRIPT_PATH)
    print(f"Found {len(segments)} raw segments (after filtering songs/English)\n")

    print("Merging short segments...")
    merged = merge_segments(segments)
    print(f"Result: {len(merged)} merged segments\n")

    print("=" * 80)
    print("PREVIEW OF MERGED SEGMENTS")
    print("=" * 80)

    short_count = 0
    long_count = 0

    for i, seg in enumerate(merged):
        duration = seg['end_sec'] - seg['start_sec']
        start_str = seconds_to_time(seg['start_sec'])
        end_str = seconds_to_time(seg['end_sec'])

        # Flag potential issues
        flag = ""
        if duration < 2:
            flag = " [SHORT]"
            short_count += 1
        elif duration > 12:
            flag = " [LONG]"
            long_count += 1

        merged_indicator = f" (merged {seg['original_count']})" if seg['original_count'] > 1 else ""

        print(f"\n[{i+1:03d}] {start_str}-{end_str} ({duration:.1f}s){merged_indicator}{flag}")
        print(f"      {seg['text'][:100]}{'...' if len(seg['text']) > 100 else ''}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(merged)} segments")
    print(f"  - Still short (<2s): {short_count}")
    print(f"  - Long (>12s): {long_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
