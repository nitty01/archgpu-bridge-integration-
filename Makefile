PYTHONPATH := .pkg:src
PY := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: help dev run test lint clean status verify stack stack-restart unified upgrade-webui stack-down integration install install-auto

help:
	@echo "Targets:"
	@echo "  make dev         - run the bridge with autoreload on :8000"
	@echo "  make run         - run the bridge (production-style, no reload)"
	@echo "  make stack       - start bridge + Open WebUI (if needed) and run full status"
	@echo "  make stack-restart  - stop both, then start + status (re-connect; see scripts/unified-stack.sh)"
	@echo "  make unified     - same as stack-restart (default: restart + connect)"
	@echo "  make upgrade-webui - pull latest Open WebUI image and recreate container (keeps data)"
	@echo "  make stack-down  - stop bridge (our pid) + Open WebUI container"
	@echo "  make verify      - headless verify (status + optional VERIFY_CHAT=1)"
	@echo "  make install     - check/install bridge dependencies (manual mode)"
	@echo "  make install-auto - install missing apt packages + build runtime image"
	@echo "  make test        - run the unit test suite"
	@echo "  make integration - run the optional integration test"
	@echo "  make status      - check bridge + Open WebUI health and wiring"
	@echo "  make clean       - remove caches"

status:
	@bash scripts/status.sh

verify:
	@bash scripts/verify-webui-bridge.sh

install:
	@bash scripts/install-bridge.sh

install-auto:
	@bash scripts/install-bridge.sh --auto-install-packages

stack:
	@bash scripts/stack.sh up

stack-restart:
	@bash scripts/stack.sh restart

unified:
	@bash scripts/unified-stack.sh

upgrade-webui:
	@bash scripts/upgrade-openwebui.sh

stack-down:
	@bash scripts/stack.sh down

dev:
	ARCHGPU_BRIDGE_RELOAD=1 ARCHGPU_BRIDGE_LOG_LEVEL=DEBUG \
	  $(PY) -m archgpu_ollama_bridge

run:
	ARCHGPU_BRIDGE_LOG_LEVEL=INFO \
	  $(PY) -m archgpu_ollama_bridge

test:
	$(PY) -m pytest -q

integration:
	$(PY) -m pytest -q -m integration

clean:
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/__pycache__
