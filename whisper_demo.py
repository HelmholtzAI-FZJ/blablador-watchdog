import os
import sys
import openai

def main(audio_path: str) -> None:
    """Transcribe an audio file using the custom OpenAI‑compatible Whisper endpoint.

    The endpoint URL and API key are taken from environment variables:
        OPENAI_API_KEY – your access token for the PTJ server
        OPENAI_BASE_URL – optional, defaults to the PTJ URL

    Example:
        $ export OPENAI_API_KEY=sk-xxxx
        $ python whisper_demo.py sample.wav
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("Error: OPENAI_API_KEY environment variable is not set\n")
        sys.exit(1)

    # Allow overriding the base URL via env; fallback to the PTJ endpoint
    base_url = os.getenv(
        "OPENAI_BASE_URL", "https://ptj.blablador.fz-juelich.de/v1/"
    )

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    try:
        with open(audio_path, "rb") as f:
            # The model name may be ignored by the custom server, but ``whisper-1``
            # follows the OpenAI API schema.
            response = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="json"
            )
        print(response.text)
    except Exception as e:
        sys.stderr.write(f"Transcription failed: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python whisper_demo.py <audio_file.wav>\n")
        sys.exit(1)
    main(sys.argv[1])
