#!/usr/bin/env python3
"""
Split Nier Replicant audio and transcript by chapters.
"""

import os
import re
import subprocess

SOURCE_DIR = "source_media"
AUDIO_FILE = "source_media/【観るゲーム】NieR Replicant ver.1.22474487139...（ニーアレプリカント）メインストーリー動画 日本語音声⧸日本語字幕 4K PC版 最高画質+ [c9IGrcYaTEM].mp3"
TRANSCRIPT_FILE = "source_media/nier_transcript.txt"
OUTPUT_DIR = "source_media/nier"

# Chapter timestamps (start time, chapter name)
CHAPTERS = [
    ("0:00:00", "00_opening"),
    ("0:13:50", "01_黒ノ病"),
    ("0:27:24", "02_白ノ書"),
    ("0:41:50", "03_謎ノ女"),
    ("0:57:13", "04_山ノ兄弟"),
    ("1:17:00", "05_誰が為ノ手紙"),
    ("1:27:42", "06_復讐ノ果て"),
    ("1:43:18", "07_仮面ノ掟"),
    ("1:55:45", "08_迷子ノ王"),
    ("2:08:25", "09_森ノ言葉"),
    ("2:50:54", "10_洋館ノ主"),
    ("3:09:44", "11_マオウ襲来"),
    ("3:41:50", "12_5年後"),
    ("3:48:07", "13_スノウホワイト"),
    ("4:04:40", "14_眠り姫ノ目覚め"),
    ("5:11:33", "15_青イ鳥"),
    ("5:27:15", "16_守るロボット"),
    ("5:49:53", "17_神樹"),
    ("6:06:21", "18_疑いに消える村"),
    ("6:41:57", "19_砂漠ノ狼"),
    ("6:57:31", "20_最後ノ旅立ち"),
    ("7:10:19", "21_最期ノ挨拶"),
    ("7:46:05", "22_Aエンド"),
    ("8:06:42", "23_Bエンド"),
    ("8:11:00", "24_Cエンド"),
    ("8:21:52", "25_Dエンド"),
    ("8:28:15", "26_Eエンド"),
]

def time_to_seconds(time_str):
    """Convert H:MM:SS or MM:SS to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0

def seconds_to_ffmpeg(seconds):
    """Convert seconds to HH:MM:SS format for ffmpeg."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_timestamp(ts):
    """Parse timestamp in MM:SS or HH:MM:SS format to seconds."""
    parts = ts.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

def parse_transcript(filepath):
    """Parse transcript into list of (start_sec, end_sec, text) tuples."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    lines = content.strip().split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # Match timestamp pattern MM:SS-MM:SS or HH:MM:SS-HH:MM:SS
        match = re.match(r'^(\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?)$', line)
        if match:
            start = parse_timestamp(match.group(1))
            end = parse_timestamp(match.group(2))

            # Get text lines until next timestamp or empty
            i += 1
            text_lines = []
            while i < len(lines):
                text_line = lines[i].strip()
                if re.match(r'^\d+:\d+(?::\d+)?-\d+:\d+(?::\d+)?$', text_line) or text_line == '':
                    break
                text_lines.append(text_line)
                i += 1

            text = ' '.join(text_lines)
            if text:
                segments.append((start, end, text))
        else:
            i += 1

    return segments

def split_audio():
    """Split audio file into chapters using ffmpeg."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, (start_time, name) in enumerate(CHAPTERS):
        start_sec = time_to_seconds(start_time)

        # Get end time (start of next chapter or end of file)
        if i + 1 < len(CHAPTERS):
            end_sec = time_to_seconds(CHAPTERS[i + 1][0])
        else:
            end_sec = None  # To end of file

        output_file = os.path.join(OUTPUT_DIR, f"{name}.mp3")

        cmd = ['ffmpeg', '-y', '-i', AUDIO_FILE, '-ss', seconds_to_ffmpeg(start_sec)]
        if end_sec:
            cmd.extend(['-to', seconds_to_ffmpeg(end_sec)])
        cmd.extend(['-c', 'copy', output_file])

        print(f"Extracting: {name} ({start_time})")
        subprocess.run(cmd, capture_output=True)

    print(f"\nAudio split into {len(CHAPTERS)} chapters in {OUTPUT_DIR}/")

def split_transcript():
    """Split transcript into chapter files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nParsing transcript...")
    segments = parse_transcript(TRANSCRIPT_FILE)
    print(f"Found {len(segments)} segments")

    # Convert chapter times to seconds
    chapter_times = [(time_to_seconds(t), name) for t, name in CHAPTERS]

    for i, (chapter_start, name) in enumerate(chapter_times):
        # Get end time
        if i + 1 < len(chapter_times):
            chapter_end = chapter_times[i + 1][0]
        else:
            chapter_end = float('inf')

        # Filter segments for this chapter
        chapter_segments = [
            (start, end, text) for start, end, text in segments
            if start >= chapter_start and start < chapter_end
        ]

        # Write chapter transcript (adjust timestamps to be relative to chapter start)
        output_file = os.path.join(OUTPUT_DIR, f"{name}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for start, end, text in chapter_segments:
                # Keep absolute timestamps for now (easier for debugging)
                rel_start = start - chapter_start
                rel_end = end - chapter_start
                start_str = f"{rel_start // 60:02d}:{rel_start % 60:02d}"
                end_str = f"{rel_end // 60:02d}:{rel_end % 60:02d}"
                f.write(f"{start_str}-{end_str}\n{text}\n\n")

        print(f"{name}: {len(chapter_segments)} segments")

    print(f"\nTranscripts split into {OUTPUT_DIR}/")

def main():
    print("=" * 60)
    print("SPLITTING NIER REPLICANT BY CHAPTERS")
    print("=" * 60)

    print("\n1. Splitting audio...")
    split_audio()

    print("\n2. Splitting transcript...")
    split_transcript()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
