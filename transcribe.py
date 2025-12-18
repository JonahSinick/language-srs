#!/usr/bin/env python3
"""
Transcribe audio files using mlx-whisper (GPU-accelerated on Apple Silicon).
Usage: python transcribe.py source_media/episode02.m4a
"""

import sys
import os
import mlx_whisper

def transcribe(audio_path, model="mlx-community/whisper-large-v3-turbo"):
    """Transcribe an audio file to text with timestamps."""

    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return

    print(f"Transcribing with MLX (GPU): {audio_path}")
    print(f"Model: {model}")

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        language="ja",
        verbose=True
    )

    # Generate output filename
    base = os.path.splitext(audio_path)[0]
    output_path = base + ".txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start_min = int(segment["start"] // 60)
            start_sec = int(segment["start"] % 60)
            end_min = int(segment["end"] // 60)
            end_sec = int(segment["end"] % 60)

            timestamp = f"{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}"
            text = segment["text"].strip()

            f.write(f"{timestamp}\n{text}\n\n")
            print(f"{timestamp}: {text}")

    print("-" * 60)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        print("Example: python transcribe.py source_media/episode02.m4a")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcribe(audio_file)
