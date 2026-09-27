"""Turn a Jarvis SSE response into ordered browser text and PCM frames."""

import asyncio
import json
import logging
import re


log = logging.getLogger(__name__)


def extract_tts_chunks(buffer: str):
    """Split complete sentences/newlines from an incomplete text remainder."""
    chunks = []
    buffer = buffer.replace("\r\n", "\n")

    while True:
        boundary = None
        newline_pos = buffer.find("\n")
        if newline_pos != -1:
            boundary = newline_pos + 1

        punctuation_match = re.search(r"[.!?…:;](?=\s|$)", buffer)
        if punctuation_match:
            punctuation_boundary = punctuation_match.end()
            boundary = (
                punctuation_boundary
                if boundary is None
                else min(boundary, punctuation_boundary)
            )

        if boundary is None:
            if len(buffer.strip()) >= 10000:
                split_at = buffer.rfind(" ", 0, 10000)
                boundary = split_at + 1 if split_at > 0 else 10000
            else:
                break

        chunk = buffer[:boundary].strip()
        buffer = buffer[boundary:]
        if chunk:
            chunks.append(chunk)

    return chunks, buffer


async def send_audio_chunk(websocket, audio: bytes):
    """Send raw PCM16 audio in a binary WebSocket frame."""
    if audio:
        await websocket.send(audio)


async def tts_worker(
    websocket,
    tts_queue: asyncio.Queue,
    synthesize_async,
    send_json,
):
    """Synthesize queued text sequentially so PCM frames keep text order."""
    while True:
        text = await tts_queue.get()
        try:
            if text is None:
                return

            text = text.strip()
            if not text:
                continue

            audio = await synthesize_async(text)
            if audio:
                log.debug("Synthesized %d audio bytes", len(audio))
                await send_audio_chunk(websocket, audio)
        except Exception:
            log.exception("Supertonic synthesis failed")
            try:
                await send_json(
                    websocket,
                    {"type": "error", "message": "Supertonic TTS failed"},
                )
            except Exception:
                pass
        finally:
            tts_queue.task_done()


async def stream_jarvis_to_tts(
    websocket,
    jarvis_session,
    *,
    synthesize_async,
    send_json,
):
    """Forward Jarvis text deltas and synthesize sentence-sized PCM chunks."""
    tts_queue = asyncio.Queue()
    tts_task = asyncio.create_task(
        tts_worker(websocket, tts_queue, synthesize_async, send_json)
    )
    text_buffer = ""

    try:
        async for raw_line in jarvis_session:
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue

            data_text = line[5:].strip()
            if data_text == "[DONE]":
                break

            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices")
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue

            token = delta.get("content")
            if not token:
                continue

            await send_json(
                websocket,
                {"type": "assistant_delta", "text": token},
            )

            text_buffer += token
            chunks, text_buffer = extract_tts_chunks(text_buffer)
            for chunk in chunks:
                await tts_queue.put(chunk)

        remaining = text_buffer.strip()
        if remaining:
            await tts_queue.put(remaining)

        await tts_queue.join()
    finally:
        await tts_queue.put(None)
        try:
            await tts_task
        except asyncio.CancelledError:
            pass

    await send_json(websocket, {"type": "assistant_done"})
