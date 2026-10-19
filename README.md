# SIGNA v2 — deletreo en LSM, offline y de bajo costo

```
Cámara ─► MediaPipe (mano dominante [+ hombros/nariz]) ─► features ─► MLP LiteRT ─► segmentador ─► texto + voz
Micrófono ─► Vosk ─► texto con acentos en pantalla
```

Requerimientos y trazabilidad: [`REQUIREMENTS.md`](REQUIREMENTS.md) · Cambios: [`CHANGELOG.md`](CHANGELOG.md)

## Estructura

| Módulo | Responsabilidad |
|---|---|
| `signa/config.py` | Todos los parámetros, en segundos y palmas/segundo. Se ajustan con `config.json` |
| `signa/features.py` | Landmarks → características (configuración, orientación, ubicación opcional). Versionado |
| `signa/vision.py` | Cámara en hilo propio, mano dominante, Pose opcional |
| `signa/segmenter.py` | Movimiento/detención → letras, letras dobles, fin de palabra |
| `signa/classifier.py` | Inferencia LiteRT con verificación de versión |
| `signa/dataset.py` | Un CSV por sesión, deshacer, carga vectorizada |
| `signa/train.py` | Evaluación por señante no visto, exportación, `meta.json`, vectores golden |
| `signa/ui.py` | Interfaz para la persona sorda (progreso, aviso de voz, acentos, pantalla circular) |
| `signa/tts.py`, `signa/stt.py` | Voz (Piper/eSpeak) y transcripción (Vosk) |
| `signa/check.py` | Diagnóstico del equipo y verificación golden |

## 1. Instalación (laptop)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements-pc.txt
```

### Modelos (en `models/`)

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# Solo si usarán ubicación respecto al cuerpo (--pose):
curl -L -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
python -m piper.download_voices es_MX-claude-high --download-dir models
# Vosk: descarguen vosk-model-small-es-0.42.zip de https://alphacephei.com/vosk/models y descompriman en models/
```

## 2. Uso

```bash
python -m signa check                          # ¿todo listo?
python -m signa collect --signer jose          # grabar (añadan --dominant Left si es zurdo, --pose para ubicación)
python -m signa train                          # evaluar + exportar
python -m signa run --listen                   # app completa
python -m signa run --listen --round --fullscreen   # en el dispositivo con pantalla circular
python -m pytest                               # pruebas automáticas (47)
```

Captura: `ESPACIO` grabar · `n`/`p` cambiar seña · `u` deshacer 10 · `ESC` salir.
App: `BACKSPACE` borrar · `c` limpiar · `h` repetir frase · `d` diagnóstico (FPS, velocidad, confianza) · `ESC`.

## 3. Cómo deletrear frente al dispositivo

- **Sostengan cada letra** un instante (~¼ s). El anillo se llena y se pone verde cuando se acepta.
- **Transición libre:** mientras la mano se mueve no se escribe nada.
- **Letra doble (LL, RR, CC):** un pequeño rebote o deslizamiento entre las dos, o pasar por reposo.
- **Fin de palabra:** bajar la mano ~1 s, o mano relajada (reposo). La palabra se dice en voz y aparece "VOZ".
- Zurdos: `"dominant_hand": "Left"` en `config.json`. Si hay dos manos visibles, se lee la dominante.

## 4. Calibración (con la app en modo `d`)

| Síntoma | Ajuste en `config.json` |
|---|---|
| Letras dobles falsas | subir `min_move_s` o `move_speed_hi` |
| No registra letras dobles con rebote | bajar `min_move_s` o `move_speed_hi` |
| Escribe letras en transiciones | subir `min_hold_s` |
| Lento para señantes con fluidez | bajar `min_hold_s` (p. ej. 0.18) |
| Confunde letras parecidas | subir `min_margin`; grabar más de ese par (ver `top_confusions` en `meta.json`) |

Ejemplo: `{"dominant_hand": "Left", "camera_source": "picam", "min_hold_s": 0.2}` (ver `config.example.json`).

## 5. Protocolo de datos

- Vocabulario: 21 letras estáticas del alfabeto manual LSM + reposo `_`. J, K, Ñ, Q, X, Z tienen trazo: fuera de alcance.
- Antes de grabar, un señante nativo o intérprete valida cada configuración. Referencia: *Manos con voz* (CONAPRED, 2011).
- Grabar con la cámara en la posición final del dispositivo, en varias sesiones con luz distinta.
- Meta: 4+ señantes × 22 clases × 150 muestras. Sumar señantes sordos mejora más que sumar muestras.
- `collect` solo guarda cuando la mano está quieta y bien detectada; nunca guarda imágenes.

## 6. Métricas para el Capítulo 4

| Qué | Dónde |
|---|---|
| Accuracy y F1-macro con señante no visto, por señante | `models/meta.json` → `metrics` |
| Matriz de confusión y reporte por letra | `reports/` |
| Pares más confundidos | `meta.json` → `top_confusions` |
| FPS y latencia (media, p95) por sesión | `reports/runtime.jsonl` |
| Facilidad de uso | encuesta SUS |

## 7. Raspberry Pi CM5

```bash
sudo apt install -y python3-picamera2 libportaudio2 espeak-ng
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
# copiar models/ desde la laptop (no se entrena en la Pi)
python -m signa check          # debe decir "golden: ok": la Pi calcula lo mismo que la laptop
python -m signa run --source picam --listen --round --fullscreen
```

Si los FPS son bajos: `"frame_width": 480, "frame_height": 360`. Con `--pose`, subir `pose_every_n`.
Micrófono INMP441 (I2S): habilitarlo con un overlay en `/boot/firmware/config.txt` y confirmar con `arecord -l`.

**Portar a otro lenguaje:** `models/golden.json` trae entradas crudas y salidas esperadas. Una
implementación en C++/otro lenguaje es correcta solo si reproduce esos números (tolerancias incluidas).
