"""
Voz de la persona oyente -> texto, offline con Vosk (RF-09).
Se silencia mientras el dispositivo habla (`mute_event`) para no transcribirse a sí mismo.
"""
import json
import logging
import os
import queue
import threading
from typing import Callable, Optional

log = logging.getLogger("signa.stt")


class VoiceListener:
    def __init__(self, model_dir: str, device: Optional[int] = None,
                 mute_event: Optional[threading.Event] = None):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Falta el modelo Vosk en {model_dir}. Ver README, sección 'Modelos'.")
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self._sd = sd
        self.samplerate = int(sd.query_devices(device, "input")["default_samplerate"])
        self.device = device
        self._rec = KaldiRecognizer(Model(model_dir), self.samplerate)
        self._mute = mute_event
        self.partial = ""
        self.final = ""
        self._q: "queue.Queue[bytes]" = queue.Queue()

    @property
    def text(self) -> str:
        return self.partial or self.final

    def _callback(self, indata, frames, time_info, status) -> None:
        if self._mute is None or not self._mute.is_set():
            self._q.put(bytes(indata))

    def run(self, on_final: Optional[Callable[[str], None]] = None) -> None:
        with self._sd.RawInputStream(samplerate=self.samplerate, blocksize=8000, device=self.device,
                                     dtype="int16", channels=1, callback=self._callback):
            while True:
                data = self._q.get()
                if self._rec.AcceptWaveform(data):
                    text = json.loads(self._rec.Result()).get("text", "")
                    if text:
                        self.final, self.partial = text, ""
                        if on_final:
                            on_final(text)
                else:
                    self.partial = json.loads(self._rec.PartialResult()).get("partial", "")

    def start(self) -> "VoiceListener":
        threading.Thread(target=self.run, daemon=True).start()
        return self
