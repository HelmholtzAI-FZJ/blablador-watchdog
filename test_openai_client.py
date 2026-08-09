#!/usr/bin/env python3
"""Test Whisper endpoint with OpenAI Python client"""
import os
import sys

from openai import OpenAI


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY env var")
        return 1

    client = OpenAI(
        api_key=api_key,
        base_url="https://ptj.blablador.fz-juelich.de/v1",
    )

    print("Listing models...")
    models = client.models.list()
    print(f"Available models: {[m.id for m in models.data]}")

    print("\nTranscribing sample.wav...")
    with open("sample.wav", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="faster-whisper-large-v3",
            file=audio_file,
        )

    print(f"Transcription: {transcription.text}")
    print("\nSUCCESS!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
