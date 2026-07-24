import os, sys, json, requests

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        sys.stderr.write('OPENAI_API_KEY not set\n')
        sys.exit(1)
    base = os.getenv('OPENAI_BASE_URL', 'https://ptj.blablador.fz-juelich.de/v1/')
    url = base.rstrip('/') + '/audio/transcriptions'
    with open('sample.wav', 'rb') as f:
        files = {'file': ('sample.wav', f, 'audio/wav')}
        data = {'model': 'whisper-1', 'language': 'en'}
        headers = {'Authorization': f'Bearer {api_key}'}
        resp = requests.post(url, headers=headers, files=files, data=data)
    print('Status:', resp.status_code)
    try:
        print('JSON:', resp.json())
    except Exception:
        print('Text:', resp.text)

if __name__ == '__main__':
    main()
