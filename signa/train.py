"""
Entrenamiento, evaluación y exportación (O4, RF-03, RNF-05/06/07).

Evaluación honesta: se deja fuera un SEÑANTE completo (o una sesión, si solo hay un señante) y se
prueba con él. La validación interna para early stopping se separa ANTES del aumento de datos,
para que copias aumentadas de una muestra no aparezcan a la vez en entrenamiento y validación.

Salidas: models/signa_mlp.tflite, models/meta.json (etiquetas, versión, métricas),
models/golden.json (vectores de verificación), reports/ (matriz de confusión, reporte por letra).
"""
import json
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from . import __version__
from .config import Config
from .dataset import Dataset, load_dataset
from .features import FEATURE_VERSION, augment_batch, extract_features, extract_features_batch, n_features

log = logging.getLogger("signa.train")
MIN_SAMPLES_PER_CLASS = 50


def _features(ds: Dataset, idx: np.ndarray, use_location: bool, rng=None, n_aug: int = 0):
    H, hands, A = ds.hand_px[idx], ds.handedness[idx], ds.anchors_px[idx]
    Hs, hs, As = [H], [hands], [A]
    for _ in range(n_aug):
        Hs.append(augment_batch(H, rng)); hs.append(hands); As.append(A)
    return extract_features_batch(np.concatenate(Hs), np.concatenate(hs), np.concatenate(As), use_location)


def _build_model(n_in: int, n_classes: int):
    import keras
    model = keras.Sequential([
        keras.Input(shape=(n_in,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _fit(ds: Dataset, idx: np.ndarray, y: np.ndarray, n_classes: int, use_location: bool,
         epochs: int, n_aug: int, seed: int):
    import keras
    from sklearn.model_selection import train_test_split
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = train_test_split(idx, test_size=0.15, stratify=y[idx], random_state=seed)
    Xtr = _features(ds, tr, use_location, rng, n_aug)
    Ytr = np.tile(y[tr], n_aug + 1)
    Xva, Yva = _features(ds, va, use_location), y[va]
    model = _build_model(Xtr.shape[1], n_classes)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs, batch_size=64,
              callbacks=[stop], verbose=0)
    return model


def _groups(ds: Dataset) -> Tuple[str, np.ndarray]:
    if len(set(ds.signers)) >= 2:
        return "señante no visto", ds.signers
    if len(set(ds.sessions)) >= 2:
        log.warning("Solo hay 1 señante: se evalúa dejando fuera sesiones (optimista).")
        return "sesión no vista", ds.sessions
    log.warning("Solo hay 1 señante y 1 sesión: split aleatorio 80/20 (muy optimista).")
    return "split aleatorio", None


def evaluate(ds: Dataset, y: np.ndarray, labels: List[str], use_location: bool, epochs: int,
             n_aug: int, seed: int) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    scheme, groups = _groups(ds)
    all_idx = np.arange(len(ds))
    if groups is None:
        tr, te = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=seed)
        folds = [("aleatorio", tr, te)]
    else:
        folds = [(g, all_idx[groups != g], all_idx[groups == g]) for g in sorted(set(groups))]

    y_true, y_pred, per_fold = [], [], {}
    for name, tr, te in folds:
        missing = set(range(len(labels))) - set(y[tr])
        if missing:
            log.warning("Fold '%s' omitido: faltan clases en entrenamiento %s", name,
                        [labels[i] for i in missing])
            continue
        model = _fit(ds, tr, y, len(labels), use_location, epochs, n_aug, seed)
        pred = np.asarray(model(_features(ds, te, use_location), training=False)).argmax(1)
        per_fold[name] = {"accuracy": round(float(accuracy_score(y[te], pred)), 4),
                          "f1_macro": round(float(f1_score(y[te], pred, average="macro")), 4)}
        log.info("  sin '%s': accuracy=%.3f F1-macro=%.3f", name, *per_fold[name].values())
        y_true += y[te].tolist(); y_pred += pred.tolist()
    if not y_true:
        raise RuntimeError("Ningún fold evaluable: cada señante/sesión debe tener todas las clases.")
    return {"scheme": scheme, "per_fold": per_fold,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "y_true": y_true, "y_pred": y_pred}


def _write_reports(cfg: Config, y_true, y_pred, labels: List[str]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
    os.makedirs(cfg.reports_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=range(len(labels)), target_names=labels,
                                   digits=3, zero_division=0)
    with open(os.path.join(cfg.reports_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    log.info("\n%s", report)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 11))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout(); fig.savefig(os.path.join(cfg.reports_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    # Pares de letras más confundidos: guía para grabar más datos o revisar con un señante
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.fill_diagonal(rates, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(rates, axis=None)[::-1], rates.shape))[0]
    return [f"{labels[i]}→{labels[j]} {rates[i, j]:.0%}" for i, j in pairs[:5] if rates[i, j] > 0]


def _export(cfg: Config, model, ds: Dataset, labels: List[str], use_location: bool, meta: Dict):
    import tensorflow as tf
    os.makedirs(os.path.dirname(cfg.classifier_model) or ".", exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(cfg.classifier_model, "wb") as f:
        f.write(converter.convert())
    with open(cfg.meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _write_golden(cfg, ds, labels, use_location)


def _write_golden(cfg: Config, ds: Dataset, labels: List[str], use_location: bool, per_label: int = 1):
    """RNF-06: entradas crudas + salidas esperadas. Si la Pi (u otro lenguaje) reproduce estos
    números, el dispositivo calcula exactamente lo mismo que se entrenó."""
    from .classifier import SignClassifier
    clf = SignClassifier(cfg.classifier_model, cfg.meta_file)
    samples = []
    for lab in labels:
        for i in np.flatnonzero(ds.labels == lab)[:per_label]:
            anchors = ds.anchors_px[i] if use_location and np.isfinite(ds.anchors_px[i]).all() else None
            feats = extract_features(ds.hand_px[i], ds.handedness[i], anchors, use_location)
            samples.append({
                "label": lab, "handedness": str(ds.handedness[i]),
                "hand_px": np.round(ds.hand_px[i], 4).tolist(),
                "anchors_px": None if anchors is None else np.round(anchors, 4).tolist(),
                "features": feats.tolist(), "probs": clf.probs(feats).tolist()})
    golden = {"feature_version": FEATURE_VERSION, "use_location": use_location,
              "atol_features": 1e-4, "atol_probs": 1e-3, "samples": samples}
    with open(cfg.golden_file, "w", encoding="utf-8") as f:
        json.dump(golden, f)


def benchmark(cfg: Config, n: int = 1000) -> float:
    from .classifier import SignClassifier
    clf = SignClassifier(cfg.classifier_model, cfg.meta_file)
    x = np.zeros(n_features(clf.use_location), np.float32)
    for _ in range(50):
        clf.probs(x)
    t0 = time.perf_counter()
    for _ in range(n):
        clf.probs(x)
    return (time.perf_counter() - t0) / n * 1000


def run(cfg: Config, use_location: bool = False, epochs: int = 300, n_aug: int = 3, seed: int = 42,
        skip_eval: bool = False) -> Dict:
    ds = load_dataset(cfg.data_dir)
    counts = ds.counts()
    log.info("%d muestras | %d clases | señantes: %s", len(ds), len(counts), sorted(set(ds.signers)))
    low = {k: v for k, v in counts.items() if v < MIN_SAMPLES_PER_CLASS}
    if low:
        log.warning("Clases con menos de %d muestras: %s", MIN_SAMPLES_PER_CLASS, low)
    if use_location and not np.isfinite(ds.anchors_px).all(axis=(1, 2)).any():
        raise ValueError("use_location requiere datos grabados con Pose (collect --pose).")

    labels = sorted(counts)
    y = np.searchsorted(labels, ds.labels)
    metrics: Dict = {}
    if not skip_eval:
        ev = evaluate(ds, y, labels, use_location, epochs, n_aug, seed)
        metrics = {k: ev[k] for k in ("scheme", "accuracy", "f1_macro", "per_fold")}
        metrics["top_confusions"] = _write_reports(cfg, ev["y_true"], ev["y_pred"], labels)
        log.info("Resultado (%s): accuracy=%.3f F1-macro=%.3f | más confundidas: %s",
                 metrics["scheme"], metrics["accuracy"], metrics["f1_macro"], metrics["top_confusions"])

    final = _fit(ds, np.arange(len(ds)), y, len(labels), use_location, epochs, n_aug, seed)
    meta = {"signa_version": __version__, "feature_version": FEATURE_VERSION,
            "use_location": use_location, "n_features": n_features(use_location), "labels": labels,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": {"samples": len(ds), "signers": len(set(ds.signers)),
                        "sessions": len(set(ds.sessions)), "per_label": counts},
            "training": {"epochs_max": epochs, "augment_copies": n_aug, "seed": seed},
            "metrics": metrics}
    _export(cfg, final, ds, labels, use_location, meta)
    ms = benchmark(cfg)
    meta["metrics"]["classifier_ms"] = round(ms, 4)
    with open(cfg.meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Exportado %s (%.0f KB) | %.3f ms por predicción", cfg.classifier_model,
             os.path.getsize(cfg.classifier_model) / 1024, ms)
    return meta
