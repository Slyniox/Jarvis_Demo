# Jarvis

> A local AI assistant and proof of concept for **speculative retrieval**:
> using a lightweight planning step to predict what external context an LLM will need and retrieve it before the main generation step.

Jarvis is a personal AI system built around local LLM inference
with llama.cpp. It combines conversational AI with optional
retrieval from a personal knowledge base and the web.

The project also serves as a proof of concept for **speculative retrieval**:
where a lightweight model predicts which sources of context will be useful for a request and retrieves them before the main reasoning model begins generation.

The goal is to reduce the latency introduced by retrieval while preserving
the benefits of tool-augmented generation.


## Architecture
```
                         ┌──────────────────┐
                         │   Browser UI     │
                         │  Chat / Voice    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       Chat Proxy          │
                    │   OpenAI-compatible API   │
                    └─────────────┬─────────────┘
                                  │
                         ┌────────▼────────┐
                         │     Planner     |
                         │   (speculative  |
                         |    retrieval)   |
                         └───┬────────┬────┘
                             │        │
                    ┌────────▼───┐ ┌──▼──────────┐
                    │ Diary / RAG│ │ Web Search  │
                    │FAISS+Rerank│ │  Firecrawl  │
                    └──────┬─────┘ └──────┬──────┘
                           │               │
                           └───────┬───────┘
                                   ▼
                          ┌────────────────┐
                          │   llama.cpp    │
                          │   Local LLM    │
                          └────────────────┘
```
The chat proxy starts the configured llama.cpp model and waits for its health endpoint before it serves client requests. The planner uses the same model endpoint. The voice proxy sends transcripts to the chat API.

## Speculative Retrieval

The core idea is to perform retrieval speculatively, before the main
generation model begins answering.

Instead of the main model spending valuable inference time deciding on how to call tools,
a lightweight planner call is made to predict what context is likely
to be needed and performs retrieval before generation starts.
Retrieved content is then embedded into the main model's context before generation begins.

In testing on compute-limited hardware, this approach reduced the time to
first useful token by up to 50%.

## Known project status : 🚧 Active development

This repository represents a working personal system with components at different maturity levels. Local machine paths, non-included data/model artifacts, optional service dependencies, and the lack of a consolidated install workflow mean another person will need to adapt configuration and prepare assets before running it. The browser UI is more self-contained than the backend. The project is published for review and continued development, not presented as a polished general-purpose installer.


## What it does

- Exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint with streamed responses.
- Offers **Normal** and **Fast** chat policies. Both use the same planner, diary retrieval, web retrieval, and model infrastructure, with different retrieval limits and prompts.
- Lets a planner decide whether a request needs diary context, web context, and/or additional model reasoning.
- Retrieves relevant diary entries from a SQLite database and FAISS index, with metadata summaries, person profiles, and reranking.
- Searches web pages through Firecrawl, then extracts, chunks, and reranks page content locally for model context.
- Provides a dependency-free browser UI with Markdown rendering, conversation history in browser `localStorage`, editable/regeneratable messages, a persistent system prompt, activity status, and cancellation of generation.
- Provides an optional hold-to-speak voice interface. It streams microphone audio to a local realtime speech-to-text service, sends final transcripts through the Fast chat pipeline, and streams generated speech back to the browser.


## Repository layout

```
backend/
  manager/       Chat API, model lifecycle, planner, configuration, retrieval orchestration
  web/
    Server/      Browser UI and small HTTP/SSE gateway
    firecrawl/   Search, fetch, extraction, chunking, and web reranking
  audio/         WebSocket voice proxy and Supertonic streaming integration
Diary/           Diary import, metadata, profile, embedding, and retrieval tools
```

The active voice entry point is `backend/audio/voice_proxy.py`.

## Services and ports

| Component | Default port | Role |
| --- | ---: | --- |
| llama.cpp | `8081` | Local model server, launched by the chat proxy |
| Chat proxy | `9000` | OpenAI-compatible chat API and shared orchestration |
| Realtime STT | `8080` | Local speech recognition service used by voice |
| Voice proxy | `9100` | Browser WebSocket, chat streaming, and TTS adapter |
| Web UI | `3000` | Browser app and same-origin API gateway |

STT and llama.cpp are external programs; their implementations are not included here.

## Requirements and current setup limits

The current checkout does not include a dependency lockfile or a turnkey installer. `backend/manager/requirements.txt` lists the Python packages imported across the API, retrieval, web, voice, and offline diary tools. The appropriate llama.cpp build, model weights, diary data/index, and realtime STT assets must be provided separately. Install the Python packages from the repository root with:

```bash
python3 -m pip install -r backend/manager/requirements.txt
```

The memory reranker is configured to use CUDA, so install a PyTorch build compatible with your CUDA setup if the default `torch` package does not support your GPU.

Before starting the chat proxy, prepare the following:

1. Python and the dependencies required by the modules you enable.
2. A compatible llama.cpp build with `llama-server` and the configured Qwen 3.6 35B A3B model and chat template.
3. Diary runtime artifacts expected by `Diary/memory_retriever.py`: `LJDB_FR.db`, `memory.index`, `memory_mapping.json`, the embedding model available locally, and any profiles used by your data.
4. A Firecrawl API key and the Firecrawl/Python dependencies for web retrieval. If web retrieval is not configured, planner requests requiring it will not work.
5. For voice only: the realtime STT service and the Supertonic 3 model/voice assets. Download the assets into the `backend/audio/tts/` folder; the configured model directory is `backend/audio/tts/assets/`.

`backend/manager/config.json` is the central configuration file for model paths, endpoints, ports, pipeline limits, web search, voice, and TTS. It currently contains host-specific model paths and a placeholder Firecrawl key. Update it for your machine before running the service. The diary retriever also expects its database and generated index files at paths relative to `Diary/`; those data files are not included in this checkout.

Diary databases, indexes, profiles, model files, and API keys are private/local data.

## Running the services

Start dependencies in order. Commands below reflect the current entry points; they assume that configuration, artifacts, and Python packages are already in place.

### 1. Start the chat API and model

```bash
cd backend/manager
uvicorn proxy:app --host 127.0.0.1 --port 9000
```

Run this from `backend/manager` so Python can import `proxy`. The proxy reads `config.json`, starts the configured llama.cpp server, waits for it to report ready, and then accepts requests on `127.0.0.1:9000`. Keep this process running while using the UI or voice service.

### 2. Start the browser UI

In another terminal:

```bash
cd backend/web/Server
python3 server.py
```

Open [http://localhost:3000](http://localhost:3000). The gateway serves the static frontend and forwards `/api/chat` requests to the chat API at `127.0.0.1:9000`. Its current host, port, and upstream URL are constants in `server.py`.

### 3. Start voice (optional)

Start the realtime speech-to-text server separately. The voice proxy documents this example:

```bash
nemo-speech serve --asr-model nemotron-3.5
```

Then, in another terminal:

```bash
cd backend/audio
python3 voice_proxy.py
```

The voice service reads its STT, chat, TTS, and port settings from `backend/manager/config.json`. Download the Supertonic 3 model and voice assets into `backend/audio/tts/assets/` before starting the proxy; the directory is not included in the repository. Voice uses the Fast pipeline and French voice instructions.

## Chat API

The main endpoint is `POST /v1/chat/completions`. It accepts an OpenAI-style `messages` array and supports `pipeline: "normal"` or `pipeline: "fast"`. The default is normal. Responses are streamed as server-sent events.

Example:

```bash
curl -N http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt",
    "pipeline": "normal",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello, Jarvis."}]
  }'
```

The stream contains OpenAI-compatible model chunks and Jarvis status events for stages such as planning, diary retrieval, web retrieval, reranking, thinking, and generation. The UI understands those status events and displays them in its activity timeline. The proxy also exposes `/v1/models` and `/health`.

The browser app stores conversations and its selected pipeline/system prompt in that browser's local storage. They are not stored by a server-side conversation database. Chat messages are sent to the local chat API; if web retrieval is selected, search queries go through Firecrawl and retrieved pages are fetched locally for context.

## Diary data preparation

The scripts under `Diary/` are offline maintenance tools as well as part of runtime retrieval. Preserve the input and generated data formats when modifying them. The common preparation flow is:

```bash
cd Diary
python3 main.py       # import LJDB.md into LJDB_FR.db
python3 metadata.py   # extract structured metadata for entries
python3 embeddings.py # create FAISS index and vector mapping
```

This flow needs the source `LJDB.md`, a running compatible local model for metadata extraction, and the required embedding/FAISS dependencies and model. `LJDB.md` and the generated diary artifacts are not included in the current checkout. Profile and candidate scripts in `Diary/` are additional offline tools.

### `LJDB.md` entry format

The parser starts a new entry only when it finds a standalone line containing a date in **day/month/year** order, followed by a colon. For example, use `01/02/2025 :` for 1 February 2025. The regular expression permits whitespace before the colon and trailing whitespace, but requires the date itself to be `DD/MM/YYYY`. Do not use `#` headings or `YYYY-MM-DD` dates: those lines will not delimit entries. All text after one date line belongs to that entry until the next date line. Blank lines are fine, and lines made only of underscores may be used as visual separators.

```markdown
01/02/2025 :

Today I started a new project. This is the first diary entry.
I can use multiple paragraphs.

____________________
02/02/2025 :

Today I continued working on the project.
```

Use valid calendar dates. Entries should be in chronological order.

## Voice protocol

The browser sends mono PCM16 little-endian audio frames and JSON control messages (`start`, `end`, and `ping`) to the voice proxy. The proxy forwards audio to the configured STT WebSocket. It forwards partial and final transcripts to the UI, sends final text to `/v1/chat/completions` with `pipeline="fast"`, and returns assistant text plus streamed mono PCM16 little-endian audio frames. The `audio_start` event includes the output sample rate and format. On plain HTTP localhost the UI connects directly to port `9100`.

## Development and verification

The current checkout does not contain a `tests/` directory. Files named `tester.py` are manual diagnostic programs, not an automated test suite.

## License

No license is currently specified in this repository. Add a license before inviting reuse or redistribution; without one, standard copyright applies.
