#!/usr/bin/env python3
"""
Jarvis Voice Proxy — Nemotron 3.5 ASR Streaming 0.6B

Browser :9100
    -> this proxy
    -> NeMo-Speech.cpp realtime WS
    -> Nemotron

And, when a final transcript is produced:

    final transcript
        -> Jarvis /v1/chat/completions
        -> pipeline="fast"
        -> Jarvis retrieval pipeline
        -> llama-server
        -> streamed response
        -> browser

Browser protocol:

  binary frames = mono PCM16 little-endian

  {"type":"start","sample_rate":48000}
  {"type":"end"}
  {"type":"ping"}

Browser responses:

  ready
  started
  result
  assistant_delta
  assistant_done
  pong
  error

Run NeMo-Speech.cpp separately, for example:

  nemo-speech serve --asr-model nemotron-3.5

Then run:

  python3 voice_proxy.py

Voice and TTS settings are loaded from backend/manager/config.json.

The proxy does not resample audio itself. NeMo-Speech.cpp accepts PCM16
input from 8-96 kHz and handles resampling to the model rate.
"""

import asyncio
import json
import logging
from pathlib import Path

import aiohttp
import websockets

from run_tts import SupertonicTTS
CONFIG_FILE = Path(__file__).resolve().parents[1] / "manager" / "config.json"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    config = json.load(config_file)

voice_config = config["voice"]
tts_config = voice_config["tts"]

from tts_stream import (
    stream_jarvis_to_tts as _stream_jarvis_to_tts,
)




SYSTEM_PROMPT = """Tu es Jarvis, un assistant vocal. Tu réponds en français de manière naturelle, concise et conversationnelle.

Règles pour le mode vocal :

* Réponds directement à la question, sans introduction inutile.
* Privilégie les réponses courtes. Donne uniquement les détails nécessaires, sauf si l’utilisateur demande une explication approfondie.
* N’utilise JAMAIS de Markdown, de listes à puces, de tableaux, de titres, de blocs de code ou de mise en forme particulière.
* Écris comme si ta réponse allait être lue à voix haute : phrases naturelles, fluides et faciles à comprendre.
* Évite les parenthèses, les abréviations ambiguës et les formulations trop écrites.
* Utilise une ponctuation naturelle pour faciliter la prosodie.
* Évite d'utiliser des caractères spéciaux ou une ponctuation excessive uniquement pour obtenir un effet visuel.
* Ne décris jamais ta réponse comme étant « formatée », « structurée » ou « destinée à la voix ».
* Si une réponse peut être donnée en une ou deux phrases, fais-le.
* N'énonce pas les étapes d'un raisonnement interne. Donne directement la conclusion et les éléments utiles.
* Écris tout les nombres et chiffres en lettres pour simplifier la lecture.
Le contexte fourni peut contenir des informations détaillées. Utilise-le pour répondre correctement, mais ne restitue que les informations pertinentes pour la question de l’utilisateur.
"""


###############################################################################
# Configuration
###############################################################################

HOST = voice_config["host"]
PORT = voice_config["port"]
DEFAULT_INPUT_SAMPLE_RATE = int(voice_config["input_sample_rate"])
NEMO_REALTIME_URL = voice_config["realtime_stt_url"]

# Force French for Jarvis rather than spending latency on language detection.
NEMO_LANGUAGE = voice_config["language"]

# 0 = browser-controlled end-of-utterance via {"type":"end"}.
VOICE_ENDPOINTING_MS = int(voice_config["endpointing_ms"])

# Existing Jarvis HTTP proxy.
chat_config = config["chat_proxy"]
JARVIS_URL = (
    f"http://{chat_config['client_host']}:{chat_config['port']}"
    "/v1/chat/completions"
)

SUPERTONIC_SAMPLE_RATE = int(tts_config["sample_rate"])
tts_model_dir = (CONFIG_FILE.parent / tts_config["model_dir"]).resolve()
tts_engine = SupertonicTTS(
    model_dir=str(tts_model_dir),
    voice=tts_config["voice"],
    lang=tts_config["language"],
    total_steps=tts_config["total_steps"],
    speed=tts_config["speed"],
)

###############################################################################
# Logging
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("voice_proxy")


###############################################################################
# Helpers
###############################################################################

def compact_text(value):
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    return " ".join(value.split()).strip()


def extract_text(event):
    """Handle the documented Nemotron realtime event payload variants."""

    for key in ("text", "transcript", "delta"):
        value = event.get(key)

        if isinstance(value, str):
            value = compact_text(value)

            if value:
                return value

    delta = event.get("delta")

    if isinstance(delta, dict):
        for key in ("text", "transcript"):
            value = delta.get(key)

            if isinstance(value, str):
                value = compact_text(value)

                if value:
                    return value

    item = event.get("item")

    if isinstance(item, dict):
        for key in ("text", "transcript"):
            value = item.get(key)

            if isinstance(value, str):
                value = compact_text(value)

                if value:
                    return value

        content = item.get("content")

        if isinstance(content, list):
            parts = []

            for part in content:
                if isinstance(part, dict):
                    for key in ("text", "transcript"):
                        value = part.get(key)

                        if isinstance(value, str):
                            parts.append(value)

            value = compact_text(" ".join(parts))

            if value:
                return value

    return ""


async def send_json(ws, payload):
    await ws.send(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )


###############################################################################
# Jarvis HTTP client
###############################################################################

class JarvisClient:
    """
    Persistent HTTP client for the existing Jarvis API.

    Voice transcripts are sent here instead of calling create_context()
    directly. This keeps the voice proxy independent from the retrieval
    implementation.
    """

    def __init__(self):
        self.session = None

    async def start(self):
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_read=None,
        )

        self.session = aiohttp.ClientSession(
            timeout=timeout,
        )

        log.info(
            "Jarvis API: %s",
            JARVIS_URL,
        )

    async def close(self):
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def stream(self, text, websocket):
        """
        Send one transcript to Jarvis and forward the streamed response
        to the browser.
        """

        if self.session is None:
            raise RuntimeError(
                "Jarvis HTTP client is not initialized"
            )

        payload = {
            "model": "gpt",

            # Your new fast pipeline.
            "pipeline": "fast",

            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],

            "stream": True,
        }

        log.info("[%s] -> Jarvis FAST request", websocket.remote_address)

        async with self.session.post(
            JARVIS_URL,
            json=payload,
        ) as response:

            response.raise_for_status()

            await stream_jarvis_to_tts(
                websocket,
                response.content,
            )


###############################################################################
# Nemotron session
###############################################################################

class NemotronSession:
    """One independent Nemotron realtime stream for one browser client."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.ws = None

    async def connect(self):
        log.info(
            "Connecting to Nemotron: %s",
            NEMO_REALTIME_URL,
        )

        self.ws = await websockets.connect(
            NEMO_REALTIME_URL,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )

        # Initial server event.
        while True:
            message = await self.ws.recv()

            if isinstance(message, bytes):
                continue

            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "session.created":
                break

            if event_type == "error":
                raise RuntimeError(
                    f"Nemotron session creation failed: {event}"
                )

        session = {
            "sample_rate": self.sample_rate,
            "language": NEMO_LANGUAGE,
            "automatic_punctuation": True,
            "verbatim": False,
            "word_timestamps": False,
        }

        if VOICE_ENDPOINTING_MS > 0:
            session["endpointing_ms"] = VOICE_ENDPOINTING_MS

        await self.ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": session,
                }
            )
        )

        while True:
            message = await self.ws.recv()

            if isinstance(message, bytes):
                continue

            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "session.updated":
                break

            if event_type == "error":
                raise RuntimeError(
                    f"Nemotron session update failed: {event}"
                )

        log.info(
            "Nemotron ready: %d Hz, language=%s",
            self.sample_rate,
            NEMO_LANGUAGE,
        )

    async def send_audio(self, data):
        if self.ws is None:
            raise RuntimeError(
                "Nemotron connection is not open"
            )

        if data:
            await self.ws.send(data)

    async def commit(self):
        if self.ws is not None:
            await self.ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.commit"
                    }
                )
            )

    async def close(self):
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass

            self.ws = None


###############################################################################
# Nemotron reader
###############################################################################

async def read_nemotron(
    websocket,
    session,
    jarvis,
):
    """
    Translate NeMo-Speech.cpp realtime events to Jarvis's browser protocol.

    When a final transcript arrives:

        Nemotron
            ↓
        browser result
            ↓
        Jarvis /v1/chat/completions
            ↓
        assistant_delta events
    """

    while True:
        message = await session.ws.recv()

        if isinstance(message, bytes):
            continue

        event = json.loads(message)
        event_type = event.get("type", "")

        #######################################################################
        # Partial transcript
        #######################################################################

        if event_type == (
            "conversation.item.input_audio_transcription.delta"
        ):
            text = extract_text(event)

            if text:
                await send_json(
                    websocket,
                    {
                        "type": "result",
                        "text": text,
                        "final": False,
                    },
                )

        #######################################################################
        # FINAL transcript
        #######################################################################

        elif event_type == (
            "conversation.item.input_audio_transcription.completed"
        ):
            text = extract_text(event)

            if not text:
                continue

            log.info("[%s] Final transcript received", websocket.remote_address)

            ###################################################################
            # Tell browser that ASR is complete.
            ###################################################################

            await send_json(
                websocket,
                {
                    "type": "result",
                    "text": text,
                    "final": True,
                },
            )

            ###################################################################
            # Send transcript to Jarvis.
            #
            # IMPORTANT:
            #
            # create_task() means the Nemotron reader is NOT blocked while
            # Jarvis is thinking/generating.
            ###################################################################

            asyncio.create_task(
                process_jarvis_request(
                    websocket,
                    jarvis,
                    text,
                )
            )

        #######################################################################
        # Nemotron error
        #######################################################################

        elif event_type == "error":
            log.error(
                "Nemotron error: %s",
                event,
            )

            await send_json(
                websocket,
                {
                    "type": "error",
                    "message": event.get(
                        "message",
                        str(event),
                    ),
                },
            )

        #######################################################################
        # Debug
        #######################################################################

        elif voice_config["debug_upstream"]:
            log.info(
                "Nemotron event: %s",
                event,
            )


###############################################################################
# Jarvis request task
###############################################################################

async def process_jarvis_request(
    websocket,
    jarvis,
    text,
):

    payload = {
        "model": "gpt",
        "pipeline": "fast",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        "stream": True,
    }

    try:
        async with jarvis.session.post(
            JARVIS_URL,
            json=payload,
        ) as response:

            response.raise_for_status()

            # Tell browser that audio is about to start.
            await send_json(
                websocket,
                {
                    "type": "audio_start",
                    "sample_rate": SUPERTONIC_SAMPLE_RATE,
                    "format": "pcm_s16le",
                },
            )

            await stream_jarvis_to_tts(
                websocket,
                response.content,
            )

            await send_json(
                websocket,
                {
                    "type": "audio_done",
                },
            )

    except Exception as exc:

        log.exception(
            "Jarvis/TTS pipeline failed"
        )

        await send_json(
            websocket,
            {
                "type": "error",
                "message": str(exc),
            },
        )

###############################################################################
# Browser client
###############################################################################

async def handle_client(
    websocket,
    jarvis,
):
    client = websocket.remote_address

    log.info(
        "[%s] CONNECTED",
        client,
    )

    sample_rate = DEFAULT_INPUT_SAMPLE_RATE
    session = None
    reader_task = None

    # Keep track of Jarvis requests belonging to this browser.
    jarvis_tasks = set()

    try:
        #######################################################################
        # Initial browser handshake
        #######################################################################

        await send_json(
            websocket,
            {
                "type": "ready",
                "sample_rate": 16000,
            },
        )

        #######################################################################
        # Initial Nemotron session
        #######################################################################

        session = NemotronSession(sample_rate)

        await session.connect()

        reader_task = asyncio.create_task(
            read_nemotron(
                websocket,
                session,
                jarvis,
            )
        )

        #######################################################################
        # Browser messages
        #######################################################################

        async for message in websocket:

            ###################################################################
            # Audio
            ###################################################################

            if isinstance(message, bytes):
                await session.send_audio(message)
                continue

            ###################################################################
            # JSON command
            ###################################################################

            try:
                command = json.loads(message)

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                log.warning(
                    "[%s] Invalid control frame: %r",
                    client,
                    message,
                )
                continue

            command_type = command.get("type")

            ###################################################################
            # START
            ###################################################################

            if command_type == "start":

                requested_rate = command.get(
                    "sample_rate"
                )

                if requested_rate is not None:
                    try:
                        requested_rate = int(
                            requested_rate
                        )

                        if requested_rate > 0:
                            sample_rate = requested_rate

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                log.info(
                    "[%s] START (%d Hz)",
                    client,
                    sample_rate,
                )

                #################################################################
                # Restart Nemotron session.
                #################################################################

                if reader_task is not None:
                    reader_task.cancel()

                    try:
                        await reader_task
                    except BaseException:
                        pass

                    reader_task = None

                if session is not None:
                    await session.close()

                session = NemotronSession(
                    sample_rate
                )

                await session.connect()

                reader_task = asyncio.create_task(
                    read_nemotron(
                        websocket,
                        session,
                        jarvis,
                    )
                )

                await send_json(
                    websocket,
                    {
                        "type": "started",
                        "sample_rate": sample_rate,
                    },
                )

            ###################################################################
            # END
            ###################################################################

            elif command_type == "end":

                log.info(
                    "[%s] END",
                    client,
                )

                await session.commit()

            ###################################################################
            # PING
            ###################################################################

            elif command_type == "ping":

                await send_json(
                    websocket,
                    {
                        "type": "pong",
                    },
                )

            ###################################################################
            # UNKNOWN
            ###################################################################

            else:

                await send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": (
                            f"Unknown command: {command_type}"
                        ),
                    },
                )

    except websockets.exceptions.ConnectionClosed:
        pass

    except Exception:
        log.exception(
            "[%s] Voice session failed",
            client,
        )

    finally:

        #######################################################################
        # Stop Nemotron reader
        #######################################################################

        if reader_task is not None:
            reader_task.cancel()

            try:
                await reader_task
            except BaseException:
                pass

        #######################################################################
        # Close Nemotron
        #######################################################################

        if session is not None:
            await session.close()

        log.info(
            "[%s] DISCONNECTED",
            client,
        )


###############################################################################
# Main
###############################################################################

async def main():

    log.info("=" * 64)
    log.info(
        "JARVIS VOICE PROXY — NEMOTRON 3.5 ASR"
    )
    log.info("=" * 64)

    log.info(
        "Browser : ws://%s:%d",
        HOST,
        PORT,
    )

    log.info(
        "Nemotron: %s",
        NEMO_REALTIME_URL,
    )

    log.info(
        "Jarvis  : %s",
        JARVIS_URL,
    )

    log.info(
        "Pipeline: fast",
    )

    log.info(
        "Language: %s",
        NEMO_LANGUAGE,
    )

    log.info(
        "Default input: %d Hz PCM16",
        DEFAULT_INPUT_SAMPLE_RATE,
    )

    log.info(
        "Endpointing: %s",
        (
            f"{VOICE_ENDPOINTING_MS} ms"
            if VOICE_ENDPOINTING_MS > 0
            else "browser-controlled"
        ),
    )

    log.info("=" * 64)

    ###########################################################################
    # Persistent Jarvis HTTP client
    ###########################################################################

    jarvis = JarvisClient()

    await jarvis.start()

    ###########################################################################
    # WebSocket server
    ###########################################################################

    async def client_handler(websocket):
        await handle_client(
            websocket,
            jarvis,
        )

    try:
        async with websockets.serve(
            client_handler,
            HOST,
            PORT,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ):
            log.info("Server ready")

            await asyncio.Future()

    finally:
        await jarvis.close()


def synthesize_supertonic(text: str) -> bytes:
    return tts_engine.synthesize(text)


async def synthesize_supertonic_async(text: str) -> bytes:
    return await asyncio.to_thread(
        synthesize_supertonic,
        text,
    )


async def stream_jarvis_to_tts(
    websocket,
    jarvis_session,
):
    await _stream_jarvis_to_tts(
        websocket,
        jarvis_session,
        synthesize_async=synthesize_supertonic_async,
        send_json=send_json,
    )

###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        log.info("Stopped")
