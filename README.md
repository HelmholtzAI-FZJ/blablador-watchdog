# blablador-watchdog
This checks if blablador is working correctly by asking each model for a word. This should ensure that the models are responding.

## Usage
```bash
./run.sh
```

## Requirements
- uv: `pip install uv`
- openai, colorama and dotenv: `uv pip install openai colorama python-dotenv flake8 pytest`

## Configuration
Create a `.env` file with the following content:
```
API_KEY=your blablador key
OPENAI_BASE_URL=https://api.helmholtz-blablador.fz-juelich.de/v1
```