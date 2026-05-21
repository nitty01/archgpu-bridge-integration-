# Chat Context Summary

This file is a compact handoff for starting a new chat without re-explaining the full history.

## Primary Goal

Build a local AI stack on Ubuntu with Intel Arc GPU acceleration, then evolve it into a smarter model-serving layer that behaves more like Ollama:

- load models on demand
- avoid keeping all models resident all the time
- expose a clean API for UI tools and app frameworks
- eventually support pull-and-deploy workflows

## User Preferences / Constraints

The user prefers:

- step-by-step execution with verification after each major step
- debugging and retries instead of blind scripting
- minimal, practical solutions
- local-first tooling
- Intel Arc GPU acceleration where possible

## Existing Working Local Stack

The earlier setup was completed in a different workspace and lives under:

- `/home/nitender-kumar/llm`

### What is already working

- `llama.cpp` running with Intel Arc GPU acceleration through SYCL
- `Open WebUI` running locally
- multiple `llama-server` instances on separate ports
- Open WebUI connected to all of them
- GPU activity verified during inference

### Services

- Open WebUI: `http://localhost:3000`
- General model API: `http://localhost:8080/v1`
- Code model API: `http://localhost:8081/v1`
- Finance model API: `http://localhost:8082/v1`

### Model IDs

- `mistral.gguf`
- `code.gguf`
- `finance.gguf`

### Open WebUI Login

- Email: `admin@local.test`
- Password: `LocalLLM!2345`

## Models Currently Used

### General

- `mistral.gguf`

### Code

- `code.gguf`
- based on a compact Qwen2.5 Coder 3B GGUF

### Finance / analysis

- `finance.gguf`
- based on a compact Qwen2.5 3B Instruct GGUF used as a finance-analysis slot

## Important Runtime Decisions Already Made

These settings were chosen because they worked reliably on the machine:

- explicit `/dev/dri` device mapping into containers
- `SYCL_DEVICE_FILTER=level_zero:gpu`
- `--cache-ram 0` to avoid severe prompt-cache stalls
- `-np 1` for stability
- one model per `llama-server` container on its own port

This means:

- model switching currently happens by choosing a different backend/port
- each model server handles one active request stream at a time
- parallel requests to the same model can queue

## Performance / GPU Notes

- GPU use was verified by observing Intel Arc frequency rise during inference
- the multi-model stack fit in GPU memory with the chosen compact models
- the code model and finance model were much more practical than adding more 7B-class servers

## Documentation Already Written

There is already a detailed usage guide here:

- `/home/nitender-kumar/llm/LOCAL_LLM_SETUP_AND_USAGE.md`

That document includes:

- local stack overview
- direct API usage
- LangGraph examples
- LangChain examples
- OpenAI SDK usage
- LlamaIndex / LiteLLM / DSPy notes
- routing patterns and troubleshooting

## Known Recent Issue

The `code.gguf` server originally ran with:

- context size `4096`

This caused a real error in Open WebUI:

- `request (4274 tokens) exceeds the available context size (4096 tokens), try increasing it`

### What was done

The code model container was restarted with:

- `-c 8192`

### Latest observed state

Logs confirmed the restarted code server loaded with:

- `n_ctx = 8192`

and still fit on GPU.

### Remaining note

The final long-prompt verification was interrupted while the server was still warming up, so if a new chat needs certainty, the first check should be:

1. confirm `llama-code` is healthy
2. confirm a >4096-token request now succeeds

## How Apps Currently Use The Stack

Apps can call the local OpenAI-compatible APIs directly:

- `http://localhost:8080/v1`
- `http://localhost:8081/v1`
- `http://localhost:8082/v1`

with:

- API key: `none`

This already works with:

- LangGraph
- LangChain
- OpenAI Python SDK
- LlamaIndex
- LiteLLM
- similar OpenAI-compatible libraries

## Important Product Idea Under Discussion

The next major idea is to build a bridge/orchestrator between UI clients and local model servers so the system behaves more like Ollama.

### Desired behavior

- models are not all kept loaded all the time
- a request for a model can trigger loading that model on demand
- inactive models can be unloaded
- the system decides when to launch or stop `llama-server`
- eventually support model pull + register + deploy workflow
- ideally compatible with Open WebUI and/or Ollama-style clients

### High-level bridge concept

Possible responsibilities of the bridge app:

- maintain a model registry
- know which GGUF files are available locally
- optionally pull GGUFs on demand
- start/stop `llama-server` containers dynamically
- proxy requests to the right backend
- present either:
  - an Ollama-compatible API
  - an OpenAI-compatible API
  - or both
- keep a small cache / eviction policy for loaded models

## Summary Of The Most Useful Next Work

If starting a fresh chat in this new project folder, the likely next task is:

Design and build an `ARCHGPU_OLLAMA_BRIDGE` application that:

- manages GGUF model lifecycle dynamically
- proxies client requests
- supports on-demand loading/unloading
- keeps Open WebUI or similar clients working smoothly

## Good Starting Assumptions For The New Chat

Unless the user changes direction, reasonable defaults are:

- build the bridge in this current repo/folder
- target the existing local models under `/home/nitender-kumar/llm/models`
- keep using `llama.cpp` containers as the execution backend
- optimize for local reliability first, not distributed scale
- support Open WebUI as an important client
- prefer an OpenAI-compatible facade first, unless Ollama compatibility is explicitly prioritized

## Short Version

The user already has a working local multi-model Intel Arc + `llama.cpp` + Open WebUI stack. The next likely project is a bridge layer that makes local GGUF models behave more like Ollama: pull/register models, load them only when needed, unload them when idle, and route requests transparently through a stable API.
