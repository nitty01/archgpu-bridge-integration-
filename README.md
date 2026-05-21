# ARCHGPU OLLAMA Bridge

Local-first Python bridge that drives `llama.cpp` SYCL backends behind stable
OpenAI-compatible and Ollama-compatible APIs, with on-demand load and idle
unload of model containers.

## What it does

- Loads models on demand by starting a `llama-server` Docker container per
  request, mapped to a stable host port.
- Pulls GGUF models directly from Hugging Face on `POST /api/pull`, persists
  them under your models directory, and registers them as dynamic models
  alongside the static ones declared in [config/models.yaml](config/models.yaml).
- Unloads idle models after a configurable TTL and enforces a max number of
  concurrently loaded models (LRU eviction).
- Exposes:
  - OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions` (with SSE
    streaming passthrough)
  - Ollama-compatible `GET /api/tags`, `POST /api/show`, `GET /api/ps`,
    `POST /api/chat`, `POST /api/generate`, `POST /api/pull`, `POST /api/delete`
    (NDJSON streaming for chat/generate/pull)
  - Health probe at `GET /health`

## Quickstart

Install dependencies (PEP 660 editable install or any package manager you
prefer), then:

```bash
make dev      # autoreload, debug logging
make run      # production-style
make test     # unit tests
make integration  # opt-in end-to-end tests against real Docker
```

### One-command: bridge + Open WebUI

From the project root, bring up the bridge (in the background) and the Open
WebUI container, then run the same checks as `make status`:

```bash
make stack
# or: ./scripts/stack.sh up
```

**One script, restart + re-connect (stop both, start again, run status):** use this when
things are already running but you want a clean cycle and the same wiring checks as `make status`.

```bash
make unified
# or: make stack-restart
# or: ./scripts/unified-stack.sh
# or: ./scripts/stack.sh restart
```

With no arguments, `./scripts/unified-stack.sh` runs **`restart`**. For a gentle start only
(if nothing is running), use `./scripts/unified-stack.sh up` or `make stack`.

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Open WebUI: [http://127.0.0.1:3000](http://127.0.0.1:3000) (by default
  `OPENWEBUI_NETWORK_MODE=bridge`: published port 3000, bridge URLs use
  `host.docker.internal` — see table below). Use `OPENWEBUI_NETWORK_MODE=host`
  for `--network host` and `127.0.0.1` in env instead.

Status only: `make status` or `./scripts/status.sh`. Headless checks (status +
optional chat to the bridge): `make verify` or
`./scripts/verify-webui-bridge.sh` (`VERIFY_CHAT=1` to also POST chat completion).
Stop what this script started: `make stack-down` or `./scripts/stack.sh down`.

**Stack / Open WebUI environment (shell, not `ARCHGPU_*`):**

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENWEBUI_NETWORK_MODE` | `bridge` | `bridge`: `-p 3000:8080`, `--add-host=host.docker.internal:host-gateway`, Ollama/OpenAI base URLs use `http://host.docker.internal:<bridge_port>`. `host`: `--network host`, URLs use `http://127.0.0.1:<bridge_port>`. |
| `OPENWEBUI_HOST_PORT` | `3000` | Host port mapped to Open WebUI when `OPENWEBUI_NETWORK_MODE=bridge`. |
| `OPENWEBUI_DATA_VOLUME` | `openwebui-data` | Docker volume for WebUI data. If `ARCHGPU_UNATTENDED_DEV=1`, defaults to `openwebui-archgpu-data`. |
| `RECREATE_OPENWEBUI` | `0` | Set to `1` to `docker rm` the container and recreate (volume kept). |
| `AUTO_RECREATE_OPENWEBUI` | `0` | Set to `1` to recreate once if in-container `GET /v1/models` still fails after startup waits. |
| `ARCHGPU_UNATTENDED_DEV` | `0` | Set to `1` for local-only unattended profile: default isolated volume, `WEBUI_AUTH=False`, `ENABLE_PERSISTENT_CONFIG=False` on **new** `docker run` (only works on a volume with **no** existing users; **do not** use in production). |

If you already had an Open WebUI container, either set **Settings → Connections** manually, or recreate: `RECREATE_OPENWEBUI=1 make stack`. For a stubborn mis-created container, `AUTO_RECREATE_OPENWEBUI=1 make stack`. See `scripts/stack.sh help` for the full list.

**Verify Open WebUI can use the bridge:** run `make status`. When Open WebUI
uses Docker’s default bridge network, it must **not** use `http://127.0.0.1:8000`
for the API (that is localhost *inside* the container). Use
`http://host.docker.internal:8000` for Ollama, or
`http://host.docker.internal:8000/v1` for the OpenAI-compatible connection.
The status script now calls `GET /v1/models` and `GET /api/tags` **from inside**
the `openwebui` container to the same host it used for `/health`—if those pass,
the same URLs will work in **Settings → Connections** once you save them there.

The bridge listens on `:8000` by default and proxies to model backends started
on per-model ports declared in [config/models.yaml](config/models.yaml).

### Pulling a model

Either pass an alias from [config/catalogue.yaml](config/catalogue.yaml) or a
full Hugging Face reference of the form `<org>/<repo>:<filename.gguf>`:

```bash
curl -N -X POST http://127.0.0.1:8000/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"name": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF:qwen2.5-coder-3b-instruct-q4_k_m.gguf"}'
```

The response is NDJSON shaped like Ollama's pull stream
(`pulling manifest`, `downloading`, `success`, then `registered`). The model
will then appear in `/api/tags` and is selectable from Open WebUI. To delete a
dynamic model and remove its file from disk:

```bash
curl -X POST http://127.0.0.1:8000/api/delete \
  -H 'Content-Type: application/json' \
  -d '{"name": "<ollama-name>"}'
```

## Configuration

All settings are environment variables prefixed with `ARCHGPU_BRIDGE_`. Common
ones:

- `ARCHGPU_BRIDGE_BACKEND_DRIVER` (`docker` or `none`)
- `ARCHGPU_BRIDGE_BACKEND_IMAGE`
- `ARCHGPU_BRIDGE_BACKEND_MODELS_HOST_DIR`
- `ARCHGPU_BRIDGE_BACKEND_DEVICES`
- `ARCHGPU_BRIDGE_IDLE_TTL_SECONDS`
- `ARCHGPU_BRIDGE_MAX_LOADED_MODELS`
- `ARCHGPU_BRIDGE_CATALOGUE_PATH` (curated pull aliases)
- `ARCHGPU_BRIDGE_DYNAMIC_MODELS_PATH` (state file for pulled models)
- `ARCHGPU_BRIDGE_DYNAMIC_PORT_RANGE` (e.g. `[18000, 18099]`)
- `ARCHGPU_BRIDGE_HF_BASE_URL`
- `ARCHGPU_BRIDGE_HF_ALLOW_ORGS` (allowlist; empty = any org)
- `ARCHGPU_BRIDGE_PULL_MAX_BYTES` (optional safety cap)

See [src/archgpu_ollama_bridge/config.py](src/archgpu_ollama_bridge/config.py)
for the full list.

## Upgrading Open WebUI (Docker)

If Open WebUI runs as a container you already have (any name), pull the latest
image and **recreate** the container with the same ports, volumes, and
environment:

```bash
export OPENWEBUI_NAME=openwebui
export OPENWEBUI_IMAGE=ghcr.io/open-webui/open-webui:main-slim   # or :main
./scripts/upgrade-openwebui.sh
# or: make upgrade-webui   (defaults: name=openwebui, image=main-slim)
```

Your data stays on the bind mount or volume; only the image is updated. To use
`main` instead of `main-slim`, set `OPENWEBUI_IMAGE` before running.

## Running as a service

A reference systemd unit lives at
[scripts/archgpu-bridge.service](scripts/archgpu-bridge.service).

## Compatibility Disclaimer

This bridge setup is validated in a local environment on Ubuntu 26.04 with:

- Intel Arc GPU (Docker device passthrough via `/dev/dri`)
- Docker Engine + Linux host networking/bridge networking
- Open WebUI container connected to the bridge

It may work on other Linux versions/distributions and hardware, but behavior can
vary with kernel/driver/runtime differences. Treat this project as a
best-effort local deployment baseline, not a production SLA.

## Publish To GitHub

If you want this project in your own GitHub repository:

```bash
cd /path/to/ARCHGPU_OLLAMA_BRIDGE

# initialize once (if .git is missing)
git init
git branch -M main
git add .
git commit -m "Initial commit: ARCHGPU OLLAMA bridge with Intel Arc support"

# set your own GitHub repo URL
git remote add origin https://github.com/<your-user>/<your-bridge-repo>.git
git push -u origin main
```

After push, keep your deployment docs updated in this README whenever:

- You change Docker image/runtime requirements
- You add or remove catalogue behavior
- You change Open WebUI integration endpoints
