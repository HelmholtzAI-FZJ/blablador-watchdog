# blablador-watchdog
This checks if blablador is working correctly by asking each model for a word. Embedding models are tested via an embedding request instead of chat. This should ensure that the models are responding.

## Usage
```bash
./run.sh
```

## Installation

```bash
# Install uv if you don't have it
pip install uv

# Create virtual environment and install dependencies
uv venv
uv sync

# Install dev dependencies (for testing/linting)
uv sync --extra dev
```

## Configuration
Create a `.env` file with the following content:
```
API_KEY=your blablador key
OPENAI_BASE_URL=https://api.helmholtz-blablador.fz-juelich.de/v1
```


## TODO

- [ ] Implement retry logic for failed requests with exponential backoff
- [ ] Add CI/CD integration (GitHub Actions) for automated testing
- [ ] Add configurable temperature parameter (currently hardcoded at 0.7)
- [ ] Expand test coverage to include more edge cases and error scenarios
- [ ] Add detailed error classification beyond CUDA/Internal Server errors
- [ ] Add JSON output format for better integration with other tools
- [ ] Add configurable timeout per model (currently global 45s)
- [ ] Add historical performance tracking over multiple runs

## Embedding models
Embedding models are routed to an embedding test (not chat) and configured in `embedding_models.py`.
