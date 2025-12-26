PY := python3
VENV_DIR := .venv
VENV311 := .venv311
ACTIVATE := $(VENV_DIR)/bin/activate
APP_PATH := $(realpath app.py)
VENV311_PY := $(VENV311)/bin/python

.PHONY: all help venv install setup run format lint test clean

.DEFAULT_GOAL := all

all: NextDown
	@echo "Built ./NextDown — run './NextDown' to start the app"

NextDown: app.py
	@echo "Creating NextDown executable..."
	@echo '#!/usr/bin/env bash' > NextDown
	@echo 'VENV311_PY="$$(pwd)/$(VENV311_PY)"' >> NextDown
	@echo 'APP_PATH="$(APP_PATH)"' >> NextDown
	@echo 'if [ -x "$$VENV311_PY" ]; then' >> NextDown
	@echo '  exec "$$VENV311_PY" -m streamlit run "$$APP_PATH" "$$@"' >> NextDown
	@echo 'else' >> NextDown
	@echo '  # Fallback to any streamlit on PATH using system python' >> NextDown
	@echo '  exec python3 -m streamlit run "$$APP_PATH" "$$@"' >> NextDown
	@echo 'fi' >> NextDown
	@chmod +x NextDown

help:
	@echo "Available targets:"
	@echo "  make           -> build ./NextDown (default)" 
	@echo "  ./NextDown     -> run the app after running make" 
	@echo "  make setup     -> create venv and install requirements"
	@echo "  make venv      -> create virtual environment"
	@echo "  make install   -> install python packages from requirements.txt"
	@echo "  make run       -> run prediction script inside venv"
	@echo "  make format    -> format code with black"
	@echo "  make lint      -> run flake8"
	@echo "  make test      -> run pytest"
	@echo "  make clean     -> remove virtualenv, caches, and NextDown"

venv:
	$(PY) -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"

install: venv
	. $(ACTIVATE) && pip install --upgrade pip
	. $(ACTIVATE) && pip install -r requirements.txt

setup: install

run: venv
	. $(ACTIVATE) && $(PY) prediction.py

format: venv
	. $(ACTIVATE) && black .

lint: venv
	. $(ACTIVATE) && flake8 .

test: venv
	. $(ACTIVATE) && pytest -q

clean:
	rm -rf $(VENV_DIR) .pytest_cache __pycache__ *.pyc NextDown
