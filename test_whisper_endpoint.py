#!/usr/bin/env python3
"""
Test the PTJ Whisper endpoint at https://ptj.blablador.fz-juelich.de/v1/
"""
import os
import sys

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
    sys.exit(1)

import httpx

BASE_URL = "https://ptj.blablador.fz-juelich.de/v1"

def test_models():
    """Test the /v1/models endpoint"""
    print(f"GET {BASE_URL}/models")
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: {resp.json()}")
            return True
        else:
            print(f"Error: {resp.text}")
            return False

def test_transcription(audio_file: str):
    """Test the /v1/audio/transcriptions endpoint"""
    print(f"\nPOST {BASE_URL}/audio/transcriptions (file={audio_file})")
    
    # Build multipart form manually to ensure correct format
    import subprocess
    import json
    
    cmd = [
        "curl", "-s", "-X", "POST",
        "-H", f"Authorization: Bearer {API_KEY}",
        "-F", f"file=@{audio_file}",
        "-F", "model=whisper-1",
        "-F", "language=en",
        f"{BASE_URL}/audio/transcriptions"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    print(f"Exit code: {result.returncode}")
    print(f"Response: {result.stdout}")
    if result.stderr:
        print(f"Stderr: {result.stderr}")
    
    # Try to parse as JSON
    try:
        data = json.loads(result.stdout)
        if result.returncode == 0 and "text" in data:
            return True
    except:
        pass
    
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PTJ Whisper Endpoint")
    print(f"URL: {BASE_URL}")
    print("=" * 60)
    
    # Test 1: List models
    models_ok = test_models()
    
    # Test 2: Transcription - use sample.wav if it exists
    audio_file = "sample.wav"
    if os.path.exists(audio_file):
        transcription_ok = test_transcription(audio_file)
    else:
        print(f"\nNo audio file found at {audio_file}, skipping transcription test")
        transcription_ok = None
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Models endpoint:      {'✓ PASS' if models_ok else '✗ FAIL'}")
    if transcription_ok is not None:
        print(f"  Transcription test:   {'✓ PASS' if transcription_ok else '✗ FAIL'}")
    print("=" * 60)
