"""Validation-only parameter selection and reusable, fitted CPU classifiers.

No test images or labels enter this module's feature extraction or fitting.
The saved pipeline consumes feature vectors; ``kind`` and ``config`` in each
bundle define the identical image-to-feature transformation used at prediction.
"""
from __future__ import annotations

import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .features import extract_features
from .labels import CLASS_NAMES


def _pipeline(name: str, value: float | int, config: dict) -> Pipeline:
    if name == "hog_knn":
        estimator = KNeighborsClassifier(
            n_neighbors=int(value), weights="distance", algorithm="brute",
            metric="euclidean", n_jobs=2,
        )
    else:
        estimator = LinearSVC(
            C=float(value), dual=False, max_iter=10000, tol=1e-4,
            random_state=int(config.get("seed", 42)),
        )
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def _fit(pipeline: Pipeline, x: np.ndarray, y: np.ndarray) -> tuple[float, list[str]]:
    """Record convergence; retry once at a larger iteration budget if needed."""
    start = time.perf_counter()
    notes = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(x, y)
    notes.extend(str(w.message) for w in caught)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        pipeline.set_params(classifier__max_iter=50000)
        notes.append("Retried fit with max_iter=50000 after convergence warning.")
        with warnings.catch_warnings(record=True) as retry:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(x, y)
        notes.extend("Retry: " + str(w.message) for w in retry)
    return time.perf_counter() - start, notes


def train_models(manifest: pd.DataFrame, config: dict) -> dict:
    """Tune three baselines on one fixed split, then refit on train + val.

    Ties in validation macro F1 retain the first candidate: ascending C or k.
    Final timings exclude features, tuning and held-out test prediction, which
    are measured separately. The scaler is fitted inside each training pipeline.
    """
    output = Path(config["output_dir"])
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pool = manifest.loc[manifest["split"].isin(["train", "val"])].copy()
    train_mask = pool["split"].to_numpy() == "train"
    val_mask = pool["split"].to_numpy() == "val"
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Parameter selection requires nonempty train and val splits.")
    labels = sorted(int(x) for x in config["classes"])
    for split in ("train", "val"):
        actual = set(pool.loc[pool["split"] == split, "label"].astype(int))
        if actual != set(labels):
            raise ValueError(f"{split} classes {sorted(actual)} differ from config {labels}.")
    y = pool["label"].to_numpy(dtype=np.int64)
    bundles, search_rows, summaries = {}, [], {}
    for kind, model_names in (("hog", ("hog_svm", "hog_knn")), ("pixels", ("pixels_svm",))):
        feature_start = time.perf_counter()
        features = extract_features(pool, config, kind=kind, cache=True)
        feature_seconds = time.perf_counter() - feature_start
        if features.ndim != 2 or len(features) != len(pool) or not np.isfinite(features).all():
            raise ValueError(f"Invalid {kind} feature matrix shape or nonfinite values.")
        x_train, x_val = features[train_mask], features[val_mask]
        for name in model_names:
            key = "n_neighbors" if name == "hog_knn" else "C"
            values = sorted(set(config.get("knn_k", [3, 5]) if name == "hog_knn"
                                else config.get("svm_C", [0.01, 0.1, 1.0])))
            if not values:
                raise ValueError(f"No parameter candidates for {name}.")
            best_score, best_value = -np.inf, None
            tuning_start = time.perf_counter()
            for value in values:
                if name == "hog_knn" and int(value) > train_mask.sum():
                    raise ValueError("KNN k exceeds the number of training images.")
                pipeline = _pipeline(name, value, config)
                fit_seconds, notes = _fit(pipeline, x_train, y[train_mask])
                predict_start = time.perf_counter()
                pred = pipeline.predict(x_val)
                predict_seconds = time.perf_counter() - predict_start
                score = f1_score(y[val_mask], pred, labels=labels, average="macro", zero_division=0)
                search_rows.append({
                    "model": name, "feature": kind, "parameter": key,
                    "value": value, "val_accuracy": accuracy_score(y[val_mask], pred),
                    "val_macro_f1": score, "fit_seconds": fit_seconds,
                    "val_predict_seconds": predict_seconds,
                    "train_samples": int(train_mask.sum()), "val_samples": int(val_mask.sum()),
                    "feature_dimension": int(features.shape[1]),
                    "max_iter": pipeline.get_params().get("classifier__max_iter", None),
                    "warnings": json.dumps(notes, ensure_ascii=False),
                })
                if score > best_score:
                    best_score, best_value = score, value
            tuning_seconds = time.perf_counter() - tuning_start
            final = _pipeline(name, best_value, config)
            refit_seconds, notes = _fit(final, features, y)
            timings = {"train_seconds": refit_seconds, "refit_seconds": refit_seconds,
                       "tuning_seconds": tuning_seconds, "feature_seconds": feature_seconds}
            selected = {key: int(best_value) if key == "n_neighbors" else float(best_value)}
            metadata = {
                "name": name, "kind": kind, "config": config, "classes": labels,
                "class_names": {i: CLASS_NAMES[i] for i in labels},
                "selected_params": selected, "timings": timings,
                "validation_macro_f1": float(best_score),
                "train_samples": int(train_mask.sum()), "val_samples": int(val_mask.sum()),
                "refit_samples": len(pool), "feature_dimension": int(features.shape[1]),
                "selection_rule": "maximum validation macro F1; ties choose smaller C or k",
                "final_fit_pool": "train+val; official test excluded",
                "svm_settings": {"dual": False, "tol": 1e-4,
                                 "max_iter": final.get_params().get("classifier__max_iter")}
                                if name != "hog_knn" else None,
                "knn_settings": {"weights": "distance", "metric": "euclidean", "n_jobs": 2}
                                if name == "hog_knn" else None,
                "fit_warnings": notes, "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
            }
            bundle = {**metadata, "pipeline": final}
            joblib.dump(bundle, model_dir / f"{name}.joblib", compress=3)
            bundles[name], summaries[name] = bundle, metadata
            pd.DataFrame(search_rows).to_csv(output / "validation_search.csv", index=False, encoding="utf-8-sig")
            (output / "model_summary.json").write_text(
                json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundles
