#!/usr/bin/env python3
"""
Build listening practice data for Final Fantasy X.
Usage: python build_ffx.py 00_ザナルカンド
       python build_ffx.py --list
"""

import re
import json
import subprocess
import os
import sys
import numpy as np
import librosa

SOURCE_FILE = "FFX Movie (Japanese Audio) [0pqRfQXkIXY]"
TRANSCRIPT_PATH = f"source_media/{SOURCE_FILE}.txt"
AUDIO_PATH = f"source_media/{SOURCE_FILE}.mp3"

# Merge settings
MIN_SEGMENT_DURATION = 3.0
MAX_GAP_TO_MERGE = 1.5
MAX_MERGED_DURATION = 12.0

# Audio endpoint detection settings
MIN_EXTENSION = 0.8
SEARCH_WINDOW = 2.5
SILENCE_THRESHOLD = 0.08
MIN_SILENCE_DURATION = 0.25
FALLBACK_BUFFER = 1.2
END_BUFFER = 0.35
START_BUFFER = 0.15
START_SEARCH_WINDOW = 1.0

DROP_TEXTS = {
    "うん",
    "ああ",
    "はぁ",
    "えっ",
    "あ",
    "はっ",
    "ふんっ!",
    "んー…",
    "おぉ!",
    "ヌー!",
}

# Max seconds a clip can extend beyond its transcript end time
MAX_CLIP_EXTENSION = 3.0

# Section definitions: (start_timestamp, end_timestamp) in seconds
# These define which portion of the transcript belongs to each section
SECTIONS = {
    "00_ザナルカンド":       ("00:00", "38:00"),
    "01_ビサイド":           ("38:00", "1:12:00"),
    "02_連絡船リキ号":       ("1:12:00", "1:18:00"),
    "03_キーリカ":           ("1:18:00", "1:41:00"),
    "04_連絡船ウィンノ号":   ("1:41:00", "1:47:00"),
    "05_ルカ":               ("1:47:00", "2:24:20"),
    "06_ミヘンセッツ街道":   ("2:24:20", "3:00:00"),
    "07_キノコ岩街道":       ("3:00:00", "3:27:00"),
    "08_ジョゼ街道":         ("3:27:00", "3:40:00"),
    "09_幻光河":             ("3:40:00", "3:52:00"),
    "10_グアドサラム":       ("3:52:00", "4:12:30"),
    "11_雷平原":             ("4:12:30", "4:24:00"),
    "12_マカラーニャの森":   ("4:24:00", "4:40:00"),
    "13_マカラーニャ寺院":   ("4:40:00", "5:05:00"),
    "14_サヌビア砂漠とホーム": ("5:05:00", "5:34:00"),
    "15_聖ベベル宮":         ("5:34:00", "5:50:00"),
    "16_浄罪の路":           ("5:50:00", "6:02:00"),
    "17_ナギ平原":           ("6:02:00", "6:34:00"),
    "18_ガガゼト山":         ("6:34:00", "7:00:00"),
    "19_ザナルカンド遺跡":   ("7:00:00", "7:30:00"),
    "20_シン":               ("7:30:00", "8:10:00"),
    "21_エンディング":       ("8:10:00", "8:50:00"),
}


def time_to_seconds(time_str):
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def seconds_to_time(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def seconds_to_ffmpeg_time(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def is_english(text):
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return ascii_letters / total_letters > 0.5


def is_noise(text):
    """Filter out sound effects, music markers, and garbled Whisper output."""
    t = text.strip()
    if t in DROP_TEXTS:
        return True
    if t.startswith("*") or t.startswith("-"):
        return True
    # Very short segments with no kanji/katakana are usually noise
    if len(t) <= 2 and not any('一' <= c <= '鿿' or '゠' <= c <= 'ヿ' for c in t):
        return True
    # YouTube artifacts and non-dialogue
    noise_patterns = [
        "ご視聴ありがとうございました",
        "音楽",
        "第1話",
    ]
    for pat in noise_patterns:
        if pat in t:
            return True
    # Al Bhed / garbled Whisper: mostly katakana gibberish with no real Japanese words
    katakana = sum(1 for c in t if 'ァ' <= c <= 'ヺ' or c == 'ー')
    hiragana = sum(1 for c in t if 'ぁ' <= c <= 'ん')
    kanji = sum(1 for c in t if '一' <= c <= '鿿')
    if len(t) > 5 and katakana > (hiragana + kanji) * 3 and hiragana + kanji < 3:
        return True
    return False


def parse_transcript(filepath, start_sec, end_sec):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        timestamp_match = re.match(r'^(\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?)$', line)
        if timestamp_match:
            seg_start = time_to_seconds(timestamp_match.group(1))
            seg_end = time_to_seconds(timestamp_match.group(2))

            i += 1
            text_lines = []
            while i < len(lines):
                text_line = lines[i].strip()
                if re.match(r'^\d+:\d+(?::\d+)?-\d+:\d+(?::\d+)?$', text_line) or text_line == '':
                    break
                text_lines.append(text_line)
                i += 1

            text = ' '.join(text_lines)

            # Only include segments within our section's time range
            if seg_start >= start_sec and seg_start < end_sec:
                if text and not is_english(text) and not is_noise(text):
                    segments.append({
                        'start_sec': seg_start,
                        'end_sec': seg_end,
                        'text': text
                    })
        else:
            i += 1

    return segments


def merge_segments(segments):
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

        should_merge = False

        if current_duration < MIN_SEGMENT_DURATION:
            if gap <= MAX_GAP_TO_MERGE:
                should_merge = True
        elif gap <= 0.5:
            potential_duration = seg['end_sec'] - current['start_sec']
            if potential_duration <= MAX_MERGED_DURATION:
                should_merge = True

        if should_merge:
            potential_duration = seg['end_sec'] - current['start_sec']
            if potential_duration <= MAX_MERGED_DURATION:
                current['end_sec'] = seg['end_sec']
                current['texts'].append(seg['text'])
            else:
                merged.append(finalize_segment(current))
                current = {
                    'start_sec': seg['start_sec'],
                    'end_sec': seg['end_sec'],
                    'texts': [seg['text']]
                }
        else:
            merged.append(finalize_segment(current))
            current = {
                'start_sec': seg['start_sec'],
                'end_sec': seg['end_sec'],
                'texts': [seg['text']]
            }

    merged.append(finalize_segment(current))
    return merged


def finalize_segment(current):
    return {
        'start_sec': current['start_sec'],
        'end_sec': current['end_sec'],
        'text': ' / '.join(current['texts'])
    }


def find_silence_after(audio, sr, start_time, max_search_time, next_start_time=None, global_rms_max=None):
    search_start_time = start_time
    start_sample = int(search_start_time * sr)
    end_sample = int((search_start_time + min(max_search_time, MAX_CLIP_EXTENSION)) * sr)

    if next_start_time is not None:
        max_end_sample = int((min(next_start_time - 0.05, start_time + MAX_CLIP_EXTENSION)) * sr)
        end_sample = min(end_sample, max_end_sample)

    end_sample = min(end_sample, len(audio))

    max_fallback = start_time + MAX_CLIP_EXTENSION
    if next_start_time is not None:
        fallback_end = min(next_start_time - 0.05, max_fallback)
    else:
        fallback_end = min(start_time + FALLBACK_BUFFER, max_fallback)

    if start_sample >= end_sample:
        return fallback_end

    search_audio = audio[start_sample:end_sample]

    if len(search_audio) < int(MIN_SILENCE_DURATION * sr):
        return fallback_end

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(y=search_audio, frame_length=frame_length, hop_length=hop_length)[0]

    if global_rms_max is not None and global_rms_max > 0:
        rms_norm = rms / global_rms_max
    elif rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        return fallback_end

    min_silence_frames = int(MIN_SILENCE_DURATION * sr / hop_length)

    for i in range(len(rms_norm) - min_silence_frames):
        if np.all(rms_norm[i:i+min_silence_frames] < SILENCE_THRESHOLD):
            silence_start_sample = i * hop_length
            result = search_start_time + (silence_start_sample / sr) + END_BUFFER
            result = max(result, start_time + MIN_EXTENSION)
            return result

    return fallback_end


def find_silence_before(audio, sr, start_time, prev_end_time=None, global_rms_max=None):
    earliest_time = 0.0
    if prev_end_time is not None:
        earliest_time = prev_end_time + 0.05

    search_start = max(earliest_time, start_time - START_SEARCH_WINDOW)
    search_end = start_time

    start_sample = int(search_start * sr)
    end_sample = int(search_end * sr)

    if start_sample >= end_sample or start_sample < 0:
        return start_time

    search_audio = audio[start_sample:end_sample]

    if len(search_audio) < int(MIN_SILENCE_DURATION * sr):
        return start_time

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(y=search_audio, frame_length=frame_length, hop_length=hop_length)[0]

    if global_rms_max is not None and global_rms_max > 0:
        rms_norm = rms / global_rms_max
    elif rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        return start_time

    min_silence_frames = int(MIN_SILENCE_DURATION * sr / hop_length)

    for i in range(len(rms_norm) - 1, min_silence_frames - 1, -1):
        if i >= min_silence_frames:
            silence_region = rms_norm[i - min_silence_frames:i]
            current_loud = rms_norm[i] >= SILENCE_THRESHOLD
            silence_before = np.all(silence_region < SILENCE_THRESHOLD)

            if current_loud and silence_before:
                speech_start_sample = i * hop_length
                new_start = search_start + (speech_start_sample / sr) - START_BUFFER
                return max(earliest_time, new_start)

    return max(earliest_time, start_time - 0.15)


def adjust_endpoints_with_audio(segments, audio, sr):
    print("   Analyzing audio for natural speech endpoints...")

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    global_rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    global_rms_max = global_rms.max()
    print(f"   Global RMS max: {global_rms_max:.4f}")

    for i, seg in enumerate(segments):
        transcript_start = seg['start_sec']
        prev_end = None
        if i > 0:
            prev_end = segments[i - 1]['end_sec']
        new_start = find_silence_before(audio, sr, transcript_start, prev_end, global_rms_max)
        seg['start_sec'] = new_start

        transcript_end = seg['end_sec']
        next_start = None
        if i < len(segments) - 1:
            next_start = segments[i + 1]['start_sec']
        new_end = find_silence_after(audio, sr, transcript_end, SEARCH_WINDOW, next_start, global_rms_max)
        seg['end_sec'] = new_end

    return segments


def split_audio_ffmpeg(segments, clips_dir, section_start, section_end):
    os.makedirs(clips_dir, exist_ok=True)

    # First extract the section's audio range to a temp file (fast single seek)
    section_audio = os.path.join(clips_dir, "_section.mp3")
    buffer = 10  # seconds of buffer on each side
    chunk_start = max(0, section_start - buffer)
    chunk_end = section_end + buffer

    print(f"   Extracting section audio chunk ({seconds_to_time(int(chunk_start))}-{seconds_to_time(int(chunk_end))})...")
    cmd = [
        'ffmpeg', '-y', '-i', AUDIO_PATH,
        '-ss', seconds_to_ffmpeg_time(chunk_start),
        '-to', seconds_to_ffmpeg_time(chunk_end),
        '-c:a', 'libmp3lame', '-q:a', '2',
        section_audio
    ]
    subprocess.run(cmd, capture_output=True)

    # Now extract clips from the small section file (fast seeks)
    for idx, seg in enumerate(segments):
        clip_filename = f"clip_{idx:03d}.mp3"
        clip_path = os.path.join(clips_dir, clip_filename)
        seg['audio_file'] = clip_filename

        # Adjust times relative to the chunk start
        start_time = seconds_to_ffmpeg_time(seg['start_sec'] - chunk_start)
        end_time = seconds_to_ffmpeg_time(seg['end_sec'] - chunk_start)

        cmd = [
            'ffmpeg', '-y', '-i', section_audio,
            '-ss', start_time,
            '-to', end_time,
            '-c:a', 'libmp3lame', '-q:a', '2',
            clip_path
        ]

        duration = seg['end_sec'] - seg['start_sec']
        print(f"[{idx+1:03d}] {seconds_to_time(seg['start_sec'])}-{seconds_to_time(int(seg['end_sec']))} ({duration:.1f}s): {seg['text'][:50]}...")
        subprocess.run(cmd, capture_output=True)

    # Clean up section audio chunk
    os.remove(section_audio)

    return segments


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_ffx.py <section_name>")
        print("       python build_ffx.py --list")
        sys.exit(1)

    if sys.argv[1] == "--list":
        print("Available sections:")
        for name, (start, end) in SECTIONS.items():
            print(f"  {name:30s}  {start} - {end}")
        sys.exit(0)

    section_name = sys.argv[1]

    if section_name not in SECTIONS:
        print(f"Unknown section: {section_name}")
        print(f"Use --list to see available sections")
        sys.exit(1)

    start_ts, end_ts = SECTIONS[section_name]
    start_sec = time_to_seconds(start_ts)
    end_sec = time_to_seconds(end_ts)

    section_dir = f"ffx/{section_name}"
    clips_dir = f"{section_dir}/clips"
    output_json = f"{section_dir}/segments.json"

    os.makedirs(section_dir, exist_ok=True)

    print("=" * 80)
    print(f"BUILDING: FFX {section_name}")
    print(f"Time range: {start_ts} - {end_ts}")
    print("=" * 80)

    print("\n1. Parsing transcript...")
    segments = parse_transcript(TRANSCRIPT_PATH, start_sec, end_sec)
    print(f"   Found {len(segments)} raw segments")

    print("\n2. Merging segments...")
    merged = merge_segments(segments)
    print(f"   Result: {len(merged)} merged segments")

    # Only load the portion of audio we need (with some buffer)
    load_start = max(0, start_sec - 5)
    load_end = end_sec + 5
    print(f"\n3. Loading audio for section ({seconds_to_time(load_start)} - {seconds_to_time(int(load_end))})...")
    audio, sr = librosa.load(AUDIO_PATH, sr=None, mono=True,
                             offset=load_start, duration=load_end - load_start)
    print(f"   Loaded {len(audio)/sr:.1f} seconds of audio at {sr}Hz")

    # Adjust segment times relative to loaded audio offset
    for seg in merged:
        seg['start_sec'] -= load_start
        seg['end_sec'] -= load_start

    print("\n4. Adjusting endpoints with audio analysis...")
    merged = adjust_endpoints_with_audio(merged, audio, sr)

    # Convert back to absolute times for ffmpeg (which reads from full file)
    for seg in merged:
        seg['start_sec'] += load_start
        seg['end_sec'] += load_start

    print("\n5. Splitting audio...")
    merged = split_audio_ffmpeg(merged, clips_dir, start_sec, end_sec)

    short = sum(1 for s in merged if s['end_sec'] - s['start_sec'] < 2)
    long_segs = sum(1 for s in merged if s['end_sec'] - s['start_sec'] > 12)
    print(f"\n   Stats: Short (<2s): {short}, Long (>12s): {long_segs}")

    print(f"\n6. Saving to {output_json}...")
    output = []
    for seg in merged:
        output.append({
            'start': seconds_to_time(seg['start_sec']),
            'end': seconds_to_time(int(seg['end_sec'])),
            'text': seg['text'],
            'audio_file': seg['audio_file']
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print(f"DONE! Created {len(merged)} clips in {clips_dir}/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
