#!/usr/bin/env python3

import time
from pathlib import Path

import numpy as np
import wave

from supertonic import TTS


class SupertonicTTS:
    """
    Persistent Supertonic TTS engine.

    The model and voice are loaded once and reused for every synthesis.

    synthesize() returns raw mono PCM16 bytes suitable for sending directly
    as a binary WebSocket frame.
    """

    def __init__(
        self,
        model_dir=None,
        voice="M1",
        lang="fr",
        total_steps=10,
        speed=1.6,
    ):
        self.model_dir = Path(model_dir or Path(__file__).resolve().parent / "tts" / "assets")

        self.voice = voice
        self.lang = lang

        self.total_steps = total_steps
        self.speed = speed

        self.tts = None
        self.voice_style = None

        self.sample_rate = None

        self._load()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):

        print("Loading Supertonic 3...", flush=True)

        start = time.perf_counter()

        self.tts = TTS(
            model_dir=str(self.model_dir),
            auto_download=False,
        )

        load_time = time.perf_counter() - start

        print(
            f"Supertonic model loaded in {load_time:.2f}s",
            flush=True,
        )

        # --------------------------------------------------------------
        # Load voice once.
        # --------------------------------------------------------------

        self.voice_style = self.tts.get_voice_style(
            voice_name=self.voice
        )

        print(
            f"Supertonic voice: {self.voice}",
            flush=True,
        )

        print(
            f"Supertonic language: {self.lang}",
            flush=True,
        )

        # --------------------------------------------------------------
        # Determine sample rate.
        #
        # Supertonic's Python API may expose this differently depending
        # on the installed version, so handle the common cases.
        # --------------------------------------------------------------

        self.sample_rate = self._get_sample_rate()

        print(
            f"Supertonic sample rate: {self.sample_rate} Hz",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Sample rate
    # ------------------------------------------------------------------

    def _get_sample_rate(self):

        # Common attribute.
        value = getattr(
            self.tts,
            "sample_rate",
            None,
        )

        if value is not None:
            return int(value)

        # Some versions expose sampling_rate.
        value = getattr(
            self.tts,
            "sampling_rate",
            None,
        )

        if value is not None:
            return int(value)

        # Fallback.
        #
        # If your installed Supertonic version has a different sample
        # rate, replace this with the actual value reported by the model.
        return 44100

    # ------------------------------------------------------------------
    # Audio conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pcm16(audio) -> bytes:
        """
        Convert Supertonic output into raw mono PCM16.

        Handles:
          float32 [-1, 1]
          float64 [-1, 1]
          int16
          numpy arrays
          torch tensors
        """

        # --------------------------------------------------------------
        # Torch tensor -> numpy
        # --------------------------------------------------------------

        if hasattr(audio, "detach"):
            audio = (
                audio
                .detach()
                .cpu()
                .numpy()
            )

        # --------------------------------------------------------------
        # Numpy
        # --------------------------------------------------------------

        audio = np.asarray(audio)

        # --------------------------------------------------------------
        # Remove unnecessary dimensions.
        #
        # Examples:
        #
        # (1, samples)
        # (samples, 1)
        # (1, samples, 1)
        # --------------------------------------------------------------

        audio = np.squeeze(audio)

        # --------------------------------------------------------------
        # If somehow still multidimensional, flatten.
        # --------------------------------------------------------------

        if audio.ndim != 1:
            audio = audio.reshape(-1)

        # --------------------------------------------------------------
        # Convert floating point audio.
        #
        # Supertonic normally returns floating-point waveform samples.
        # --------------------------------------------------------------

        if np.issubdtype(
            audio.dtype,
            np.floating,
        ):

            # Prevent clipping.
            audio = np.clip(
                audio,
                -1.0,
                1.0,
            )

            # float -> PCM16
            audio = (
                audio * 32767.0
            ).astype(
                np.int16
            )

        # --------------------------------------------------------------
        # Already PCM16.
        # --------------------------------------------------------------

        elif audio.dtype != np.int16:

            audio = audio.astype(
                np.int16
            )

        # --------------------------------------------------------------
        # Ensure little-endian PCM16.
        # --------------------------------------------------------------

        audio = audio.astype(
            "<i2",
            copy=False,
        )

        return audio.tobytes()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text and return raw PCM16 bytes.

        This function is synchronous because Supertonic inference itself
        is synchronous. The voice proxy should call it through
        asyncio.to_thread().
        """

        text = text.strip()

        if not text:
            return b""

        start = time.perf_counter()

        wav, duration = self.tts.synthesize(
            text=text,
            voice_style=self.voice_style,
            lang=self.lang,
            total_steps=self.total_steps,
            speed=self.speed,
        )

        inference_time = (
            time.perf_counter() - start
        )

          # Convert to numpy BEFORE PCM conversion
        audio = np.asarray(wav)

        audio_duration = float(
            duration[0]
        )

        pcm16 = self._to_pcm16(wav)

        rtf = (
            inference_time / audio_duration
            if audio_duration > 0
            else 0
        )

        print(
            f"[Supertonic] "
            f"{len(text)} chars | "
            f"{audio_duration:.2f}s audio | "
            f"{inference_time:.2f}s inference | "
            f"{1 / rtf:.2f}x realtime",
            flush=True,
        )

        return pcm16
