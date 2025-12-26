# NextDownAPP

Project containing `prediction.py` and data `theOne.csv`.

Quick start (macOS / zsh):

1. Create a Python virtual environment and install dependencies:

```bash
make setup
```

2. Run the prediction script:

```bash
make run
```

Useful targets:

- `make venv` — create virtualenv at `.venv`
- `make install` — install packages from `requirements.txt`
- `make format` — run `black` on the project
- `make lint` — run `flake8`
- `make test` — run `pytest`
- `make clean` — remove `.venv` and caches

Notes:

- `prediction.py` is currently empty. Add your script logic and required packages to `requirements.txt`.
- If you prefer an existing environment, activate it manually and run `pip install -r requirements.txt`.
