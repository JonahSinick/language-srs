#!/usr/bin/env python3
"""
Build SRS deck for Charlotte's Web (Spanish) with audio-based endpoint detection.
Uses transcript start times but finds natural speech endings via audio analysis.
"""

import re
import json
import subprocess
import os
import numpy as np
import librosa

TRANSCRIPT_PATH = "source_media/Charlotte's Web In Spanish Chapter 3 [5GvucfNDmgo].txt"
AUDIO_PATH = "source_media/Charlotte's Web In Spanish Chapter 3 [5GvucfNDmgo].mp3"
CLIPS_DIR = "charlottes_web/chapter_3/clips"
OUTPUT_JSON = "charlottes_web/chapter_3/segments.json"

# Merge settings
MIN_SEGMENT_DURATION = 3.0
MAX_GAP_TO_MERGE = 1.5
MAX_MERGED_DURATION = 15.0  # Slightly higher for narration

# Audio endpoint detection settings (matched to NieR settings)
MIN_EXTENSION = 0.5  # Minimum extension past transcript end before searching for silence
SEARCH_WINDOW = 2.0  # Search for silence within this window
SILENCE_THRESHOLD = 0.10  # Lower = more conservative (less likely to cut speech)
MIN_SILENCE_DURATION = 0.15  # Require longer silence to be sure it's actually silence
FALLBACK_BUFFER = 0.7  # Buffer to add if no silence found
END_BUFFER = 0.20  # Buffer after detected silence at end
START_BUFFER = 0.10  # Buffer before detected silence at start
START_SEARCH_WINDOW = 1.0  # How far back to search for start silence

# Segments to DROP (too short/filler)
DROP_TEXTS = {
    "Y.",
    "Eso.",
    "Por.",
    "Comillas.",
    "Dijo.",
    "El juego.",
    "Tío.",
    "Agatín.",
    "Side.",
    "Lado.",
    "Mediante.",
    "Dijo el.",
    "Ganso.",
    "Él.",
    "A él.",
    "Dándole.",
    "Cabriole.",
}

def time_to_seconds(time_str):
    """Convert MM:SS format to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])

def seconds_to_time(seconds):
    """Convert seconds to MM:SS format."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"

def seconds_to_ffmpeg_time(seconds):
    """Convert seconds to HH:MM:SS.ms format for ffmpeg."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def should_drop(text):
    """Check if segment should be dropped."""
    return text.strip() in DROP_TEXTS or len(text.strip()) < 3

def parse_transcript(filepath):
    """Parse the Charlotte's Web transcript format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # Remove line number prefix if present (e.g., "123→")
        if '→' in line:
            line = line.split('→', 1)[1]

        # Match timestamp format: MM:SS-MM:SS
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
                # Stop if we hit another timestamp or empty line
                if re.match(r'^(\d{2}:\d{2})-(\d{2}:\d{2})$', text_line) or text_line == '':
                    break
                text_lines.append(text_line)
                i += 1

            text = ' '.join(text_lines)

            # Filter out empty, too short, or dropped segments
            if text and not should_drop(text):
                segments.append({
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'text': text
                })
        else:
            i += 1

    return segments

def merge_segments(segments):
    """Merge short segments together intelligently."""
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

        # Merge if current segment is too short
        if current_duration < MIN_SEGMENT_DURATION:
            if gap <= MAX_GAP_TO_MERGE:
                should_merge = True
            elif gap <= 3.0 and current_duration < 1.0:
                should_merge = True
        # Also merge if gap is very small
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
    """Finalize a segment by joining text."""
    return {
        'start_sec': current['start_sec'],
        'end_sec': current['end_sec'],
        'text': ' / '.join(current['texts'])
    }

def find_silence_after(audio, sr, start_time, max_search_time, next_start_time=None, global_rms_max=None):
    """
    Find where speech ends by looking for silence after the transcript endpoint.
    Uses global_rms_max for normalization so background music doesn't confuse detection.
    """
    # Start searching from transcript end to catch speech that extends slightly
    search_start_time = start_time
    start_sample = int(search_start_time * sr)
    end_sample = int((search_start_time + max_search_time) * sr)

    if next_start_time is not None:
        max_end_sample = int((next_start_time - 0.05) * sr)
        end_sample = min(end_sample, max_end_sample)

    end_sample = min(end_sample, len(audio))

    # Fallback when we can't find silence
    if next_start_time is not None:
        fallback_end = next_start_time - 0.05  # Extend to just before next segment
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

    # Normalize against global max (whole audio) instead of local max
    # This way background music doesn't look like "loud" relative to itself
    if global_rms_max is not None and global_rms_max > 0:
        rms_norm = rms / global_rms_max
    elif rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        return fallback_end

    min_silence_frames = int(MIN_SILENCE_DURATION * sr / hop_length)

    # Find where speech drops to silence (speech-to-silence transition)
    for i in range(len(rms_norm) - min_silence_frames):
        if np.all(rms_norm[i:i+min_silence_frames] < SILENCE_THRESHOLD):
            silence_start_sample = i * hop_length
            result = search_start_time + (silence_start_sample / sr) + END_BUFFER
            # Ensure we extend at least MIN_EXTENSION past transcript end
            result = max(result, start_time + MIN_EXTENSION)
            return result

    return fallback_end

def find_silence_before(audio, sr, start_time, prev_end_time=None, global_rms_max=None):
    """Search backward from transcript start to find where speech actually begins."""
    # Don't search earlier than the previous segment's end (or 0)
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

    # Normalize against global max for consistency
    if global_rms_max is not None and global_rms_max > 0:
        rms_norm = rms / global_rms_max
    elif rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        return start_time

    min_silence_frames = int(MIN_SILENCE_DURATION * sr / hop_length)

    # Search backward from end of search region to find last silence before speech
    # We want to find where speech STARTS (end of silence)
    for i in range(len(rms_norm) - 1, min_silence_frames - 1, -1):
        # Check if this position is the END of a silence region
        if i >= min_silence_frames:
            silence_region = rms_norm[i - min_silence_frames:i]
            current_loud = rms_norm[i] >= SILENCE_THRESHOLD
            silence_before = np.all(silence_region < SILENCE_THRESHOLD)

            if current_loud and silence_before:
                # Found transition from silence to speech
                speech_start_sample = i * hop_length
                new_start = search_start + (speech_start_sample / sr) - START_BUFFER
                return max(earliest_time, new_start)

    # No clear silence-to-speech transition found, use original with small buffer
    return max(earliest_time, start_time - 0.15)

def adjust_endpoints_with_audio(segments, audio, sr):
    """Adjust segment endpoints using audio analysis."""
    print("   Analyzing audio for natural speech endpoints...")

    # Compute global RMS max for normalization
    # This ensures background music doesn't confuse silence detection
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    global_rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    global_rms_max = global_rms.max()
    print(f"   Global RMS max: {global_rms_max:.4f}")

    for i, seg in enumerate(segments):
        # Adjust START time (search backward for silence-to-speech transition)
        transcript_start = seg['start_sec']
        prev_end = None
        if i > 0:
            prev_end = segments[i - 1]['end_sec']
        new_start = find_silence_before(audio, sr, transcript_start, prev_end, global_rms_max)
        seg['start_sec'] = new_start

        # Adjust END time (search forward for speech-to-silence transition)
        transcript_end = seg['end_sec']
        next_start = None
        if i < len(segments) - 1:
            next_start = segments[i + 1]['start_sec']
        new_end = find_silence_after(audio, sr, transcript_end, SEARCH_WINDOW, next_start, global_rms_max)
        seg['end_sec'] = new_end

    return segments

def split_audio_ffmpeg(segments):
    """Split audio file into individual clips using ffmpeg."""
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
        text_preview = seg['text'][:50] + '...' if len(seg['text']) > 50 else seg['text']
        print(f"[{idx+1:03d}] {seconds_to_time(seg['start_sec'])}-{seconds_to_time(int(seg['end_sec']))} ({duration:.1f}s): {text_preview}")
        subprocess.run(cmd, capture_output=True)

    return segments

def main():
    print("=" * 80)
    print("BUILDING CHARLOTTE'S WEB CHAPTER 3 (SPANISH) SRS DECK")
    print("=" * 80)

    print("\n1. Parsing transcript...")
    segments = parse_transcript(TRANSCRIPT_PATH)
    print(f"   Found {len(segments)} raw segments (after filtering)")

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

    # Count stats
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
