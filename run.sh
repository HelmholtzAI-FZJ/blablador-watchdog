#!/bin/bash

source venv/bin/activate
OPENAI_BASE_URL=https://api.helmholtz-blablador.fz-juelich.de/v1 python main.py

OPENAI_BASE_URL=http://134.94.199.27:8080/v1 python main.py
