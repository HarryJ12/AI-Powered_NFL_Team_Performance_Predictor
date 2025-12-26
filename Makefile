PY := python3
VENV := .venv
VENV_PY := $(VENV)/bin/python
APP := app.py

.PHONY: all venv install run clean
.DEFAULT_GOAL := all

all: NextDown

NextDown: $(APP)
	@echo '#!/usr/bin/env bash' > NextDown
	@echo 'exec "$(VENV_PY)" -m streamlit run "$(APP)" "$$@"' >> NextDown
	@chmod +x NextDown

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(VENV_PY) -m pip install -r requirements.txt

run: venv
	$(VENV_PY) -m streamlit run $(APP)

clean:
	rm -rf $(VENV) NextDown