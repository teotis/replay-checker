PYTHON ?= python3

.PHONY: preflight test sync-agents

preflight:
	$(PYTHON) tools/project.py check

test:
	$(PYTHON) -m pytest

sync-agents:
	$(PYTHON) tools/project.py sync-agents
