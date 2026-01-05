#!/usr/bin/env python3
"""
Build SRS deck for Nier Automata chapters.
Processes transcript and audio for specified chapter time ranges.
"""

import re
import json
import subprocess
import os
import sys

TRANSCRIPT_PATH = "source_media/nier_automata_full.txt"
AUDIO_PATH = "source_media/nier_automata_full.mp3"
BASE_DIR = "nier_automata"

# Chapter definitions: (folder_name, start_time, end_time, display_name)
CHAPTERS = [
    # A Route
    ("a01_プロローグ", "00:00:00", "00:34:30", "A Ch.01 プロローグ"),
    ("a02_レジスタンスキャンプ", "00:34:30", "00:45:22", "A Ch.02 レジスタンスキャンプ"),
    ("a03_アダムとイヴ", "00:45:22", "01:00:55", "A Ch.03 アダムとイヴ"),
    ("a04_狂気の歌姫", "01:00:55", "01:14:59", "A Ch.04 狂気の歌姫"),
    ("a05_穿たれた大地", "01:14:59", "01:33:28", "A Ch.05 穿たれた大地"),
    ("a06_森の王様", "01:33:28", "01:50:27", "A Ch.06 森の王様"),
    ("a07_彷徨える子供", "01:50:27", "02:11:47", "A Ch.07 彷徨える子供"),
    ("a08_複製された街", "02:11:47", "02:26:18", "A Ch.08 複製された街"),
    ("a09_狂った宗教", "02:26:18", "02:46:03", "A Ch.09 狂った宗教"),
    ("a10_喪失", "02:46:03", "02:57:39", "A Ch.10 喪失"),
    ("a_end_flowers", "02:57:39", "03:03:53", "A End flowers for m[A]chines"),

    # B Route
    ("b01_プロローグ", "03:03:53", "03:33:49", "B Ch.01 プロローグ"),
    ("b02_レジスタンスキャンプ", "03:33:49", "03:36:26", "B Ch.02 レジスタンスキャンプ"),
    ("b03_アダムとイヴ", "03:36:26", "03:52:12", "B Ch.03 アダムとイヴ"),
    ("b04_狂気の歌姫", "03:52:12", "04:09:58", "B Ch.04 狂気の歌姫"),
    ("b05_穿たれた大地", "04:09:58", "04:28:22", "B Ch.05 穿たれた大地"),
    ("b06_森の王様", "04:28:22", "04:46:51", "B Ch.06 森の王様"),
    ("b07_彷徨える子供", "04:46:51", "05:12:38", "B Ch.07 彷徨える子供"),
    ("b08_複製された街", "05:12:38", "05:21:31", "B Ch.08 複製された街"),
    ("b09_狂った宗教", "05:21:31", "05:43:03", "B Ch.09 狂った宗教"),
    ("b10_喪失", "05:43:03", "05:55:53", "B Ch.10 喪失"),
    ("b_end_or_not", "05:55:53", "06:02:49", "B End or not to [B]e"),

    # C Route
    ("c11_総攻撃", "06:02:49", "06:44:31", "C Ch.11 総攻撃"),
    ("c12_肉の箱", "06:44:31", "06:55:16", "C Ch.12「肉の箱」"),
    ("c13_砂の記憶", "06:55:16", "07:06:33", "C Ch.13 砂の記憶"),
    ("c14_魂の箱", "07:06:33", "07:16:13", "C Ch.14「魂の箱」"),
    ("c15_パスカルの絶望", "07:16:13", "07:47:41", "C Ch.15 パスカルの絶望"),
    ("c16_神の箱", "07:47:41", "07:57:17", "C Ch.16「神の箱」"),
    ("c17_塔", "07:57:17", "08:51:28", "C Ch.17「塔」"),
    ("c_end_meaningless", "08:51:28", "08:55:57", "C End meaningless [C]ode"),

    # D Route
    ("d_end_childhood", "08:55:57", "09:10:00", "D End childhoo[D]'s end"),
]

# Merge settings
MIN_SEGMENT_DURATION = 3.0
MAX_GAP_TO_MERGE = 1.5
MAX_MERGED_DURATION = 12.0

# Segments to drop
DROP_TEXTS = {
    "はぁ",
    "じゃ",
}

DROP_IF_ISOLATED = {
    "はい",
    "そう",
}


def hms_to_seconds(time_str):
    """Convert HH:MM:SS to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def time_to_seconds(time_str):
    """Convert MM:SS to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def seconds_to_time(seconds):
    """Convert seconds to MM:SS format."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format for display."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def seconds_to_ffmpeg_time(seconds):
    """Convert seconds to ffmpeg time format."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def is_english(text):
    """Check if text is predominantly English."""
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return ascii_letters / total_letters > 0.5


def should_drop(text):
    """Check if segment should be dropped."""
    return text.strip() in DROP_TEXTS


def parse_transcript(filepath, chapter_start_sec, chapter_end_sec):
    """Parse transcript and extract segments within the chapter time range."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Match timestamp pattern MM:SS-MM:SS or HH:MM:SS-HH:MM:SS
        timestamp_match = re.match(r'^(\d{1,2}:\d{2}(?::\d{2})?)-(\d{1,2}:\d{2}(?::\d{2})?)$', line)
        if timestamp_match:
            start_time_str = timestamp_match.group(1)
            end_time_str = timestamp_match.group(2)
            start_sec = time_to_seconds(start_time_str)
            end_sec = time_to_seconds(end_time_str)

            i += 1
            text_lines = []
            while i < len(lines):
                text_line = lines[i].strip()
                if re.match(r'^(\d{1,2}:\d{2}(?::\d{2})?)-(\d{1,2}:\d{2}(?::\d{2})?)$', text_line) or text_line == '':
                    break
                text_lines.append(text_line)
                i += 1

            text = ' '.join(text_lines)

            # Check if within chapter range
            if start_sec >= chapter_start_sec and start_sec < chapter_end_sec:
                if text and not is_english(text) and not should_drop(text):
                    segments.append({
                        'start_sec': start_sec,
                        'end_sec': end_sec,
                        'text': text,
                        'start_sec_relative': start_sec - chapter_start_sec,
                        'end_sec_relative': end_sec - chapter_start_sec,
                    })
        else:
            i += 1

    return segments


def merge_segments(segments):
    """Merge short segments together."""
    if not segments:
        return []

    merged = []
    current = {
        'start_sec': segments[0]['start_sec'],
        'end_sec': segments[0]['end_sec'],
        'start_sec_relative': segments[0]['start_sec_relative'],
        'end_sec_relative': segments[0]['end_sec_relative'],
        'texts': [segments[0]['text']]
    }

    for i in range(1, len(segments)):
        seg = segments[i]
        current_duration = current['end_sec'] - current['start_sec']
        gap = seg['start_sec'] - current['end_sec']

        should_merge = False

        if current_duration < MIN_SEGMENT_DURATION:
            if gap <= MAX_GAP_TO_MERGE:
                should_merge = True
            elif gap <= 3.0 and current_duration < 1.0:
                should_merge = True
        elif gap <= 0.5:
            potential_duration = seg['end_sec'] - current['start_sec']
            if potential_duration <= MAX_MERGED_DURATION:
                should_merge = True

        if should_merge:
            potential_duration = seg['end_sec'] - current['start_sec']
            if potential_duration <= MAX_MERGED_DURATION:
                current['end_sec'] = seg['end_sec']
                current['end_sec_relative'] = seg['end_sec_relative']
                current['texts'].append(seg['text'])
            else:
                merged.append(finalize_segment(current))
                current = {
                    'start_sec': seg['start_sec'],
                    'end_sec': seg['end_sec'],
                    'start_sec_relative': seg['start_sec_relative'],
                    'end_sec_relative': seg['end_sec_relative'],
                    'texts': [seg['text']]
                }
        else:
            if len(current['texts']) == 1 and current['texts'][0].strip() in DROP_IF_ISOLATED:
                current_duration = current['end_sec'] - current['start_sec']
                if current_duration < 1.0:
                    current = {
                        'start_sec': seg['start_sec'],
                        'end_sec': seg['end_sec'],
                        'start_sec_relative': seg['start_sec_relative'],
                        'end_sec_relative': seg['end_sec_relative'],
                        'texts': [seg['text']]
                    }
                    continue

            merged.append(finalize_segment(current))
            current = {
                'start_sec': seg['start_sec'],
                'end_sec': seg['end_sec'],
                'start_sec_relative': seg['start_sec_relative'],
                'end_sec_relative': seg['end_sec_relative'],
                'texts': [seg['text']]
            }

    # Handle last segment
    if len(current['texts']) == 1 and current['texts'][0].strip() in DROP_IF_ISOLATED:
        current_duration = current['end_sec'] - current['start_sec']
        if current_duration >= 1.0:
            merged.append(finalize_segment(current))
    else:
        merged.append(finalize_segment(current))

    return merged


def finalize_segment(current):
    """Finalize a merged segment."""
    return {
        'start_sec': current['start_sec'],
        'end_sec': current['end_sec'],
        'start_sec_relative': current['start_sec_relative'],
        'end_sec_relative': current['end_sec_relative'],
        'text': ' / '.join(current['texts'])
    }


def split_audio_ffmpeg(segments, chapter_folder, audio_path):
    """Split audio file into individual clips."""
    clips_dir = os.path.join(BASE_DIR, chapter_folder, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    for idx, seg in enumerate(segments):
        clip_filename = f"clip_{idx:03d}.mp3"
        clip_path = os.path.join(clips_dir, clip_filename)
        seg['audio_file'] = clip_filename

        # Add small buffer for natural sound
        start_time = seg['start_sec']
        end_time = seg['end_sec'] + 0.5  # Add buffer at end

        start_ffmpeg = seconds_to_ffmpeg_time(start_time)
        end_ffmpeg = seconds_to_ffmpeg_time(end_time)

        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', start_ffmpeg,
            '-to', end_ffmpeg,
            '-acodec', 'libmp3lame',
            '-q:a', '2',
            clip_path
        ]

        duration = seg['end_sec'] - seg['start_sec']
        text_preview = seg['text'][:40] + "..." if len(seg['text']) > 40 else seg['text']
        print(f"  [{idx+1:03d}] {seconds_to_hms(seg['start_sec'])} ({duration:.1f}s): {text_preview}")
        subprocess.run(cmd, capture_output=True)

    return segments


def build_chapter(folder_name, start_time, end_time, display_name):
    """Build a single chapter."""
    start_sec = hms_to_seconds(start_time)
    end_sec = hms_to_seconds(end_time)

    print(f"\n{'='*60}")
    print(f"Building: {display_name}")
    print(f"Time range: {start_time} - {end_time}")
    print(f"{'='*60}")

    # Parse transcript
    print("\n1. Parsing transcript...")
    segments = parse_transcript(TRANSCRIPT_PATH, start_sec, end_sec)
    print(f"   Found {len(segments)} raw segments")

    if not segments:
        print("   No segments found for this chapter!")
        return

    # Merge segments
    print("\n2. Merging segments...")
    merged = merge_segments(segments)
    print(f"   Result: {len(merged)} merged segments")

    # Split audio
    print("\n3. Creating audio clips...")
    merged = split_audio_ffmpeg(merged, folder_name, AUDIO_PATH)

    # Save segments.json
    output_path = os.path.join(BASE_DIR, folder_name, "segments.json")
    output = []
    for seg in merged:
        output.append({
            'start': seconds_to_hms(seg['start_sec']),
            'end': seconds_to_hms(seg['end_sec']),
            'text': seg['text'],
            'audio_file': seg['audio_file']
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n4. Saved {len(merged)} segments to {output_path}")

    return len(merged)


def main():
    if len(sys.argv) > 1:
        # Build specific chapter by folder name
        target = sys.argv[1]
        for folder, start, end, name in CHAPTERS:
            if folder == target or target in folder:
                build_chapter(folder, start, end, name)
                return
        print(f"Chapter not found: {target}")
        print("\nAvailable chapters:")
        for folder, _, _, name in CHAPTERS:
            print(f"  {folder}: {name}")
    else:
        # Build all chapters
        print("Building all Nier Automata chapters...")
        print("="*60)

        total_segments = 0
        for folder, start, end, name in CHAPTERS:
            count = build_chapter(folder, start, end, name)
            if count:
                total_segments += count

        print(f"\n{'='*60}")
        print(f"DONE! Created {total_segments} total segments across all chapters")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
