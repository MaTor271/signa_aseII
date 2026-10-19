# Cambios v1 → v2

## Defectos corregidos

| # | Defecto en v1 | Impacto | Corrección v2 |
|---|---|---|---|
| 1 | Ventanas medidas en cuadros | Aceptar una letra tardaba 233 ms a 30 FPS y 875 ms a 8 FPS: en la Pi, deletreo ~4× más lento | Todo en segundos: `min_hold_s` ± 1 cuadro a cualquier FPS (`test_fps_independence`) |
| 2 | Letra doble solo bajando la mano | Antinatural para un señante | Rebote/deslizamiento breve o paso por reposo (`test_double_letter_*`) |
| 3 | Sin noción de movimiento | Letras basura en transiciones | Segmentación movimiento-detención (Liddell y Johnson) |
| 4 | Se elegía la mano con más "confianza" | Con dos manos visibles podía leer la mano no dominante | Mano dominante configurable (RF-02) |
| 5 | Validación tomada de datos ya aumentados | Copias de una muestra en entrenamiento y validación: early stopping engañoso | Se separa validación antes de aumentar |
| 6 | `cv2.putText` | "¿Cómo estás?" se veía con signos `?`: la persona sorda no podía leer bien al oyente | Texto con Pillow, acentos y ñ |
| 7 | Sin versión de características | Un modelo viejo con código nuevo fallaba en silencio | `meta.json` + `ModelMismatchError` |
| 8 | Captura en movimiento | Cuadros de transición etiquetados como letra | `collect` solo graba en detención y con detección fuerte |
| 9 | Un solo CSV, sin deshacer | Una sesión mala contaminaba todo | Un CSV por sesión + deshacer (`u`) |
| 10 | Carga con `iterrows` y aumento en bucle | Lento con miles de muestras | Carga y aumento vectorizados |
| 11 | Lectura de cámara bloqueante | FPS limitados por la cámara | Cámara en hilo propio, sin reprocesar cuadros |
| 12 | Salto del detector | Podía dejar la lectura "en movimiento" demasiado tiempo | Velocidad acotada (`test_recovers_quickly_after_detector_jump`) |
| 13 | El micrófono podía transcribir la voz del propio dispositivo | Texto falso para la persona sorda | Micrófono silenciado mientras habla |

## Nuevo

- **Retroalimentación para la persona sorda:** letra candidata con anillo de progreso, esqueleto por color de fase, aviso "VOZ", modo pantalla circular.
- **Ubicación opcional** respecto al cuerpo (Pose): base para vocabulario más allá del deletreo.
- **Margen de decisión:** si dos letras parecidas empatan (p. ej. M/N), no se escribe nada.
- **Fin de palabra por reposo**, además de bajar la mano.
- `python -m signa check`: diagnóstico + verificación golden (misma salida en laptop y Pi).
- `REQUIREMENTS.md` con trazabilidad requerimiento → objetivo → módulo → prueba; 47 pruebas automáticas.
- Métricas de ejecución (`reports/runtime.jsonl`) y `top_confusions` para el Capítulo 4.
- Características ~10 % más rápidas que v1 (36 µs vs 40 µs por mano) pese a calcular más.
