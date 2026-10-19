"""
Voz offline (RF-07). Piper (voz neuronal) y, si falta, eSpeak NG. Habla en un hilo aparte para no
congelar el video. `is_speaking` sirve para dos cosas: la UI muestra "VOZ" a la persona sorda, y el
micrófono deja de escuchar para no transcribir la propia voz del dispositivo.
"""
import logging
import os
import queue
import shutil
import subprocess
import threading

log = logging.getLogger("signa.tts")


class Speaker:
    def __init__(self, piper_voice: str, espeak_voice: str = "es-419"):
        self.espeak_voice = espeak_voice
        self._voice = None
        self._espeak = shutil.which("espeak-ng")
        if os.path.exists(piper_voice):
            from piper import PiperVoice
            self._voice = PiperVoice.load(piper_voice)
            log.info("Voz: Piper (%s)", os.path.basename(piper_voice))
        elif self._espeak:
            log.warning("Voz: Piper no encontrado, usando eSpeak NG")
        else:
            log.warning("Sin motor de voz: solo texto")
        self._q: "queue.Queue[str]" = queue.Queue()
        self.is_speaking = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text: str) -> None:
        if text.strip():
            self._q.put(text.lower())  # minúsculas: "HOLA" se lee como palabra, no como sigla

    def wait(self) -> None:
        """Bloquea hasta que se haya dicho todo lo encolado."""
        self._q.join()

    def _worker(self) -> None:
        while True:
            text = self._q.get()
            self.is_speaking.set()
            try:
                self._speak(text)
            except Exception:  # noqa: BLE001 - un fallo de audio nunca debe tumbar la app
                log.exception("Error de audio")
            finally:
                self.is_speaking.clear()
                self._q.task_done()

    def _speak(self, text: str) -> None:
        if self._voice is not None:
            import numpy as np
            import sounddevice as sd
            chunks = list(self._voice.synthesize(text))
            if chunks:
                sd.play(np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate)
                sd.wait()
        elif self._espeak:
            subprocess.run([self._espeak, "-v", self.espeak_voice, "-s", "150", text], check=False)
        else:
            log.info("[VOZ] %s", text)
