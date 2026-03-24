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

## Cluster Usage Monitoring

The `monitor-cluster.sh` script provides comprehensive monitoring of cluster usage, including active users, rate limits, throttling status, and model usage analytics.

**Key features:**
- Automatic port-forwarding (no manual setup needed)
- User status classification (superuser, internal, external, etc.)
- Multiple output formats (text and JSON)
- Detailed diagnostics and error handling

See the [full documentation](docs/monitor-cluster.md) for more details.

### Quick Start

```bash
# Run the script (auto-starts port-forward if needed)
./monitor-cluster.sh

# Get a summary
./monitor-cluster.sh summary

# View active users
./monitor-cluster.sh active-users --limit 10
```
