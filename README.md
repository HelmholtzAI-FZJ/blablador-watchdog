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


## TODO 

- [ ] Spawn a test for each model so github give me a better explanation for the failure
- [ ] Differenciate between a model that is answering bullshit with a gpu failure.
- [ ] Try with lowest temperature so model always answer the same
- [ ] Make sure that all "base" models are working: `alias-fast`, `alias-large`, `alias-code` and `alias-embeddings`