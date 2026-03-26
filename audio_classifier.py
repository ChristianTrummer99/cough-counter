"""
Audio capture and cough detection using MediaPipe YAMNet.

Audio is processed ephemerally — no audio data is written to disk.
"""

import queue
import threading
import numpy as np
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio as mp_audio
from mediapipe.tasks.python.components.containers import AudioData

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 15600          # ~0.975 s — YAMNet's expected window size
COUGH_SCORE_THRESHOLD = 0.35   # minimum confidence to count as a cough
DEBOUNCE_SECONDS = 1.5         # ignore repeat detections within this window


class CoughDetector:
    """
    Captures microphone audio and pushes 'cough' events onto result_queue.
    The classifier is built once and reused across multiple start/stop cycles.
    """

    def __init__(self, model_path: str, result_queue: queue.Queue):
        self.model_path = model_path
        self.result_queue = result_queue

        self._running = threading.Event()
        self._classifier = None
        self._stream = None

        # Per-session state, reset on each start()
        self._timestamp_ms = 0
        self._buffer = np.array([], dtype=np.float32)
        self._last_cough_time = 0.0
        self._buffer_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self):
        """Load and compile the TFLite model. Call once at app startup."""
        options = mp_audio.AudioClassifierOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.audio.RunningMode.AUDIO_STREAM,
            max_results=10,
            score_threshold=0.1,   # broad; we filter more strictly in callback
            result_callback=self._mediapipe_callback,
        )
        self._classifier = mp_audio.AudioClassifier.create_from_options(options)

    def start(self):
        """Begin capturing audio. build() must have been called first."""
        if self._running.is_set():
            return

        self._last_cough_time = 0.0
        with self._buffer_lock:
            self._buffer = np.array([], dtype=np.float32)

        self._running.set()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=2048,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        """Stop capturing audio."""
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def close(self):
        """Release the classifier's native thread pool. Call on app exit."""
        self.stop()
        if self._classifier is not None:
            self._classifier.close()
            self._classifier = None

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice calls this on its internal audio thread."""
        if not self._running.is_set():
            return

        mono = indata[:, 0].astype(np.float32)

        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, mono])

            while len(self._buffer) >= CHUNK_SAMPLES:
                chunk = self._buffer[:CHUNK_SAMPLES]
                self._buffer = self._buffer[CHUNK_SAMPLES:]

                audio_data = AudioData.create_from_array(chunk, SAMPLE_RATE)
                self._classifier.classify_async(audio_data, self._timestamp_ms)
                # Advance timestamp by exact chunk duration (ms)
                self._timestamp_ms += int(CHUNK_SAMPLES / SAMPLE_RATE * 1000)

    def _mediapipe_callback(self, result, timestamp_ms: int):
        """MediaPipe calls this synchronously from within classify_async."""
        if not self._running.is_set() or not result.classifications:
            return

        import time
        for category in result.classifications[0].categories:
            if (
                "cough" in category.category_name.lower()
                and category.score >= COUGH_SCORE_THRESHOLD
            ):
                now = time.monotonic()
                if now - self._last_cough_time > DEBOUNCE_SECONDS:
                    self._last_cough_time = now
                    self.result_queue.put({"event": "cough"})
                break
