#!/usr/bin/env python3
"""Test Whisper endpoint with OpenAI Python client"""
import os
import sys

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: Set OPENAI_API_KEY env var")
    sys.exit(1)

from openai import OpenAI

client = OpenAI(
    api_key=api_key,
    base_url="https://ptj.blablador.fz-juelich.de/v1"
)

# List models
print("Listing models...")
models = client.models.list()
print(f"Available models: {[m.id for m in models.data]}")

# Transcribe
print("\nTranscribing sample.wav...")
with open("sample.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )

print(f"Transcription: {transcription.text}")
print("\nSUCCESS!")
