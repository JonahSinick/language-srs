#!/usr/bin/env python3
"""
Build listening practice data for Evangelion episodes.
Usage: python build_evangelion.py 01
"""

import re
import json
import subprocess
import os
import sys
import numpy as np
import librosa

# Will be set based on command line argument
EPISODE_NUM = None
TRANSCRIPT_PATH = None
AUDIO_PATH = None
CLIPS_DIR = None
OUTPUT_JSON = None

# Merge settings
MIN_SEGMENT_DURATION = 3.0
MAX_GAP_TO_MERGE = 1.5
MAX_MERGED_DURATION = 12.0

# Audio endpoint detection settings (tuned for anime with background music)
MIN_EXTENSION = 0.7      # Longer minimum extension for trailing speech
SEARCH_WINDOW = 2.0
SILENCE_THRESHOLD = 0.15 # Higher threshold - anime has constant BGM
MIN_SILENCE_DURATION = 0.15
FALLBACK_BUFFER = 0.9    # More generous fallback
END_BUFFER = 0.35        # More buffer after detected silence
START_BUFFER = 0.10
START_SEARCH_WINDOW = 1.0

# Segments to DROP (common filler)
DROP_TEXTS = {
    "うん",
    "ああ",
    "はぁ",
    "えっ",
}

# Skip opening theme (first ~1:30 of each episode)
SKIP_BEFORE_SECONDS = 90  # Skip segments that start before 1:30

def time_to_seconds(time_str):
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

def seconds_to_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
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

def should_drop(text):
    return text.strip() in DROP_TEXTS

def parse_transcript(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Match timestamp pattern MM:SS-MM:SS or HH:MM:SS-HH:MM:SS
        timestamp_match = re.match(r'^(\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?)$', line)
        if timestamp_match:
            start_sec = time_to_seconds(timestamp_match.group(1))
            end_sec = time_to_seconds(timestamp_match.group(2))

            i += 1
            text_lines = []
            while i < len(lines):
                text_line = lines[i].strip()
                if re.match(r'^\d+:\d+(?::\d+)?-\d+:\d+(?::\d+)?$', text_line) or text_line == '':
                    break
                text_lines.append(text_line)
                i += 1

            text = ' '.join(text_lines)

            if text and not is_english(text) and not should_drop(text) and start_sec >= SKIP_BEFORE_SECONDS:
                segments.append({
                    'start_sec': start_sec,
                    'end_sec': end_sec,
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
    end_sample = int((search_start_time + max_search_time) * sr)

    if next_start_time is not None:
        max_end_sample = int((next_start_time - 0.05) * sr)
        end_sample = min(end_sample, max_end_sample)

    end_sample = min(end_sample, len(audio))

    if next_start_time is not None:
        fallback_end = next_start_time - 0.05
    else:
        fallback_end = start_time + FALLBACK_BUFFER

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

def split_audio_ffmpeg(segments):
    os.makedirs(CLIPS_DIR, exist_ok=True)

    for idx, seg in enumerate(segments):
        clip_filename = f"clip_{idx:03d}.mp3"
        clip_path = os.path.join(CLIPS_DIR, clip_filename)
        seg['audio_file'] = clip_filename

        start_time = seconds_to_ffmpeg_time(seg['start_sec'])
        end_time = seconds_to_ffmpeg_time(seg['end_sec'])

        cmd = [
            'ffmpeg', '-y', '-i', AUDIO_PATH,
            '-ss', start_time,
            '-to', end_time,
            '-c:a', 'libmp3lame', '-q:a', '2',
            clip_path
        ]

        duration = seg['end_sec'] - seg['start_sec']
        print(f"[{idx+1:03d}] {seconds_to_time(seg['start_sec'])}-{seconds_to_time(int(seg['end_sec']))} ({duration:.1f}s): {seg['text'][:40]}...")
        subprocess.run(cmd, capture_output=True)

    return segments

def main():
    global EPISODE_NUM, TRANSCRIPT_PATH, AUDIO_PATH, CLIPS_DIR, OUTPUT_JSON

    if len(sys.argv) < 2:
        print("Usage: python build_evangelion.py <episode_number>")
        print("Example: python build_evangelion.py 01")
        sys.exit(1)

    EPISODE_NUM = sys.argv[1]
    TRANSCRIPT_PATH = f"source_media/episode{EPISODE_NUM}.txt"
    AUDIO_PATH = f"source_media/episode{EPISODE_NUM}.m4a"
    CLIPS_DIR = f"evangelion/episode_{EPISODE_NUM}/clips"
    OUTPUT_JSON = f"evangelion/episode_{EPISODE_NUM}/segments.json"

    # Create output directory
    os.makedirs(f"evangelion/episode_{EPISODE_NUM}", exist_ok=True)

    print("=" * 80)
    print(f"BUILDING: Evangelion Episode {EPISODE_NUM}")
    print("=" * 80)

    print("\n1. Parsing transcript...")
    segments = parse_transcript(TRANSCRIPT_PATH)
    print(f"   Found {len(segments)} raw segments")

    print("\n2. Merging segments...")
    merged = merge_segments(segments)
    print(f"   Result: {len(merged)} merged segments")

    print("\n3. Loading audio for analysis...")
    audio, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
    print(f"   Loaded {len(audio)/sr:.1f} seconds of audio at {sr}Hz")

    print("\n4. Adjusting endpoints with audio analysis...")
    merged = adjust_endpoints_with_audio(merged, audio, sr)

    print("\n5. Splitting audio...")
    merged = split_audio_ffmpeg(merged)

    short = sum(1 for s in merged if s['end_sec'] - s['start_sec'] < 2)
    long = sum(1 for s in merged if s['end_sec'] - s['start_sec'] > 12)
    print(f"\n   Stats: Short (<2s): {short}, Long (>12s): {long}")

    print(f"\n6. Saving to {OUTPUT_JSON}...")
    output = []
    for seg in merged:
        output.append({
            'start': seconds_to_time(seg['start_sec']),
            'end': seconds_to_time(int(seg['end_sec'])),
            'text': seg['text'],
            'audio_file': seg['audio_file']
        })

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print(f"DONE! Created {len(merged)} clips in {CLIPS_DIR}/")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
