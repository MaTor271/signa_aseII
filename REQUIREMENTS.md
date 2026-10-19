# Especificación de requerimientos del software — SIGNA v2

Formato basado en ISO/IEC/IEEE 29148. Cada requerimiento tiene un ID, un criterio de aceptación
verificable, el objetivo específico de SIGNA al que responde (O1–O5) y dónde se implementa y prueba.

**Objetivos:** O1 Arquitectura · O2 Hardware · O3 Percepción · O4 Datos y modelo IA · O5 Energía y validación.

## 1. Usuarios y contexto de uso

| Usuario | Necesidad | Implicación de diseño |
|---|---|---|
| Persona sorda señante (LSM) | Ser entendida sin depender de un intérprete; saber qué está "diciendo" el dispositivo por ella | No puede oír la voz de salida → **toda la retroalimentación hacia ella es visual**: letra candidata, progreso, texto confirmado, aviso de que se emitió voz |
| Persona sorda zurda | Mismo desempeño que una persona diestra | Mano dominante configurable; la mano izquierda se refleja a forma de derecha |
| Persona oyente sin LSM | Entender y responder | Recibe voz; responde hablando y su voz aparece como texto grande, con acentos y ñ |
| Equipo de desarrollo | Iterar rápido y medir | Configuración central, pruebas automáticas, métricas exportables |

## 2. Fundamento lingüístico (LSM)

Las lenguas de señas articulan cada seña con cinco parámetros: **configuración** (forma de la mano),
**orientación** de la palma, **ubicación** en el espacio o el cuerpo, **movimiento** y **rasgos no
manuales** (rostro, mirada, torso). El deletreo manual alterna fases de **detención** (la letra) y de
**movimiento** (la transición), como describe el modelo movimiento-detención de Liddell y Johnson.

| Parámetro | v2 | Cómo |
|---|---|---|
| Configuración | Sí | 21 puntos de la mano + distancias entre yemas |
| Orientación | Sí | No se normaliza la rotación |
| Ubicación | Opcional | Muñeca relativa a hombros y nariz (Pose) con `use_pose` |
| Movimiento | Para segmentar | La velocidad separa letra y transición; habilita letras dobles. Letras con trazo (J, K, Ñ, Q, X, Z) siguen fuera de alcance |
| No manuales | No | Limitación declarada |

## 3. Requerimientos funcionales

| ID | Requerimiento | Criterio de aceptación | Obj. | Implementación · Prueba |
|---|---|---|---|---|
| RF-01 | Capturar video de webcam USB o cámara CSI | `python -m signa check` abre la cámara y reporta resolución | O1, O3 | `vision.Camera` · manual |
| RF-02 | Detectar la mano **dominante** del señante | Con dos manos visibles se usa la configurada; con una sola se usa esa | O3 | `vision.pick_hand` · `test_pick_dominant_when_two_hands` |
| RF-03 | Clasificar 21 letras estáticas del alfabeto manual LSM + reposo | F1-macro reportado con señante no visto | O4 | `classifier`, `train` · `test_pipeline.py` |
| RF-04 | Segmentar el deletreo en movimiento y detención | Una transición no produce letras; una detención ≥ `min_hold_s` produce una letra | O4 | `segmenter` · `test_spelling_word_with_transitions`, `test_transition_frames_never_become_letters` |
| RF-05 | Aceptar letras dobles con el gesto natural (rebote o deslizamiento breve) | "LL" con un rebote ≥ `min_move_s` produce dos L; sostener una L no la repite | O4 | `segmenter` · `test_double_letter_with_bounce`, `test_sustained_letter_is_not_repeated` |
| RF-06 | Terminar palabra al bajar la mano o sostener reposo | Ausencia ≥ `word_end_absent_s` o reposo ≥ `word_end_rest_s` emite la palabra | O4 | `segmenter` · `test_word_ends_with_rest`, `test_word_ends_when_hand_drops` |
| RF-07 | Pronunciar cada palabra terminada con voz en español | La palabra entra a la cola de voz sin bloquear el video | O1 | `tts.Speaker` · manual |
| RF-08 | Retroalimentación visual a la persona sorda | Pantalla muestra candidata + progreso, texto confirmado e indicador "VOZ" | O5 | `ui` · `test_ui_renders_accents_and_feedback` + prueba con usuarios |
| RF-09 | Transcribir la voz de la persona oyente | Texto parcial y final visibles, con acentos y ñ | O1 | `stt`, `ui` · `test_ui_renders_accents_and_feedback` |
| RF-10 | Corregir: borrar letra, limpiar, repetir frase | Teclas o botones táctiles mapeados a `backspace/clear/say_all` | O5 | `app`, `segmenter` · `test_backspace_and_clear` |
| RF-11 | Capturar dataset por sesión, solo en detención | Archivo por sesión; muestras en movimiento descartadas; deshacer últimas N | O4 | `collect`, `dataset` · `test_session_undo_and_reload` |

## 4. Requerimientos no funcionales

| ID | Requerimiento | Criterio de aceptación | Obj. | Implementación · Prueba |
|---|---|---|---|---|
| RNF-01 | 100 % offline | Ningún módulo abre conexiones de red en ejecución | O1 | todos · revisión |
| RNF-02 | Privacidad por diseño | Nunca se guardan imágenes ni video; solo coordenadas | O1 | `dataset` · `test_dataset_stores_only_coordinates` |
| RNF-03 | Comportamiento independiente de FPS | Una letra se acepta a `min_hold_s` ± 1 cuadro, a cualquier FPS (v1: 233 ms a 30 FPS vs 875 ms a 8 FPS) | O2, O5 | `segmenter` · `test_fps_independence` |
| RNF-04 | Latencia | p95 por cuadro ≤ 100 ms en CM5 sin Pose *(meta a validar)* | O2, O5 | `app` → `reports/runtime.jsonl` · `test_app_loop_without_hardware` |
| RNF-05 | Precisión | F1-macro ≥ 0.85 con señante no visto *(meta a validar)* | O4, O5 | `train` → `models/meta.json` |
| RNF-06 | Consistencia laptop ↔ Pi | Los vectores *golden* se reproducen con error < 1e-4 en el dispositivo | O1, O4 | `check` · `test_golden_reproduces` + `python -m signa check` en la Pi |
| RNF-07 | Compatibilidad modelo ↔ código | Un modelo con otra versión de características se rechaza con mensaje claro | O4 | `classifier` · `test_version_mismatch_is_rejected` |
| RNF-08 | Inclusión de personas zurdas | Mano izquierda reflejada produce las mismas características | O3 | `features` · `test_mirror_left_hand_equals_right` |
| RNF-09 | Accesibilidad visual | Texto ≥ 28 px, alto contraste, sin depender de audio para el señante | O5 | `ui` · `test_ui_renders_accents_and_feedback` + prueba con usuarios |
| RNF-10 | Mantenibilidad | Parámetros en `Config`; pruebas automáticas pasan con `pytest` | O1 | `config` · `test_config_overrides_and_validation`, `pytest` |
| RNF-11 | Eficiencia | Características < 50 µs por mano; carga de datos vectorizada | O2 | `features`, `dataset` · `test_feature_speed` |

## 5. Fuera de alcance (declarado)

Señas con movimiento propio (J, K, Ñ, Q, X, Z y vocabulario dinámico), rasgos no manuales, gramática
espacial (referentes en el espacio, verbos direccionales) y traducción español → LSM (avatar).
La arquitectura deja preparados los datos de ubicación y la segmentación por movimiento como base.

## 6. Validación con usuarios

Toda configuración de letra se valida con un señante nativo o intérprete antes de grabar. Las pruebas
de uso se evalúan con SUS y tasa de letras correctas por letras intentadas.
