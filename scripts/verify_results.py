"""Independently audit saved experiment tables and delivery assets.

Run after a complete experiment: python scripts/verify_results.py --output outputs/full
This reads trusted local joblib models; never use it on untrusted model files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("hog_svm", "hog_knn", "pixels_svm")


def scores(y, pred, classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=classes, average="macro", zero_division=0)
    return dict(accuracy=float(accuracy_score(y, pred)), macro_precision=float(precision),
                macro_recall=float(recall), macro_f1=float(f1))


class Audit:
    def __init__(self):
        self.checks = []
        self.notes = []

    def check(self, name, condition, detail=None):
        self.checks.append(dict(check=name, passed=bool(condition), detail=detail))

    def phase(self, name, function):
        try:
            function()
        except Exception as exc:
            self.check(name, False, f"{type(exc).__name__}: {exc}")


def verify(output: Path):
    audit = Audit()
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    classes = [int(i) for i in config["classes"]]
    manifest = pd.read_csv(output / "manifest.csv")
    provenance_file = output / "manifest_with_exclusions.csv"
    provenance = pd.read_csv(provenance_file) if provenance_file.exists() else manifest
    metric_table = pd.read_csv(output / "metrics.csv")
    active = manifest.loc[manifest.split.isin(["train", "val", "test"])]
    fit_pool = manifest.loc[manifest.split.isin(["train", "val"])]
    excluded = provenance.loc[provenance.split.eq("excluded")]
    test = manifest.loc[manifest.split.eq("test")].reset_index(drop=True)

    def partitions():
        audit.check("manifest_has_active_splits_and_optional_exclusions",
                    {"train", "val", "test"}.issubset(set(manifest.split))
                    and set(manifest.split).issubset({"train", "val", "test", "excluded"}))
        audit.check("manifest_unique_image_paths", not manifest.path.duplicated().any())
        retained_provenance = provenance.loc[provenance.split.isin(["train", "val", "test"])]
        audit.check("retained_manifest_matches_exclusion_provenance",
                    manifest.path.tolist() == retained_provenance.path.tolist()
                    and manifest.split.tolist() == retained_provenance.split.tolist())
        audit.check("provenance_unique_image_paths", not provenance.path.duplicated().any())
        train = manifest.loc[manifest.split.eq("train")]
        val = manifest.loc[manifest.split.eq("val")]
        audit.check("train_validation_tracks_disjoint", not (set(train.track) & set(val.track)))
        audit.check("official_test_paths_disjoint", not (set(fit_pool.path) & set(test.path)))
        for split, part in active.groupby("split"):
            audit.check(f"{split}_class_coverage", set(part.label.astype(int)) == set(classes))
        # The official test set is retained. Whole training physical-sign tracks
        # matching test bytes or decoded RGB are excluded, never individual frames.
        for field in ("sha256", "pixel_sha256"):
            audit.check(f"{field}_complete", provenance[field].notna().all()
                        and provenance[field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all())
            cross = active.groupby(field).split.nunique()
            audit.check(f"active_{field}_cross_split_duplicates_absent", not cross.gt(1).any())
            audit.check(f"fit_test_{field}_disjoint", not (set(fit_pool[field]) & set(test[field])))
        original_training = provenance.loc[provenance.split.ne("test")]
        duplicate_rows = original_training.loc[
            original_training.sha256.isin(test.sha256)
            | original_training.pixel_sha256.isin(test.pixel_sha256)]
        expected_excluded_tracks = set(duplicate_rows.track)
        audit.check("excluded_tracks_exactly_match_detected_test_duplicates",
                    set(excluded.track) == expected_excluded_tracks,
                    dict(expected_tracks=sorted(expected_excluded_tracks),
                         excluded_tracks=sorted(set(excluded.track))))
        expected_excluded_paths = set(original_training.loc[
            original_training.track.isin(expected_excluded_tracks), "path"])
        audit.check("duplicate_tracks_excluded_in_their_entirety", set(excluded.path) == expected_excluded_paths)
        audit.check("excluded_tracks_absent_from_fit", not (set(excluded.track) & set(fit_pool.track)))
        audit.check("excluded_paths_absent_from_fit", not (set(excluded.path) & set(fit_pool.path)))
        if len(excluded):
            exclusion_csv = pd.read_csv(output / "excluded_training_tracks.csv")
            audit.check("exclusion_csv_exactly_matches_excluded_rows",
                        set(exclusion_csv.path) == set(excluded.path)
                        and len(exclusion_csv) == len(excluded))
            audit.check("exclusion_report_marks_excluded_split", exclusion_csv.split.eq("excluded").all())
            reason = "entire_training_track_contains_exact_image_matching_official_test"
            audit.check("excluded_rows_have_explicit_reason", excluded.exclusion_reason.eq(reason).all())
            audit.check("excluded_rows_have_original_fit_split", excluded.original_split.isin(["train", "val"]).all())
            audit.check("exclusion_report_reasons_match", exclusion_csv.exclusion_reason.eq(reason).all())
        data_audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
        audit.check("audit_exclusion_count_matches_manifest", data_audit.get("excluded_rows", 0) == len(excluded))
        audit.check("audit_excluded_tracks_match_manifest", sorted(data_audit.get("excluded_tracks", []))
                    == sorted(set(excluded.track)))
        audit.check("audit_retained_counts_match_manifest", data_audit["counts"] == active.groupby("split").size().to_dict())
        if "original_split" in provenance:
            original_test = provenance.loc[provenance.original_split.eq("test")]
            audit.check("all_original_official_test_rows_retained",
                        original_test.path.tolist() == test.path.tolist()
                        and original_test.split.eq("test").all())
            audit.check("audit_original_test_count_unchanged",
                        data_audit["counts_before_decontamination"]["test"] == len(test))
            audit.check("audit_counts_with_exclusions_match_provenance",
                        data_audit["counts_including_excluded"] == provenance.groupby("split").size().to_dict())
        audit.notes.append(dict(exclusion_policy="retain official test; exclude whole matching training track",
                                excluded_images=len(excluded), excluded_tracks=sorted(set(excluded.track))))
        audit.check("metrics_contains_three_models", set(metric_table.model) == set(MODELS)
                    and len(metric_table) == len(MODELS))

    audit.phase("partition_validation", partitions)
    predictions = {}

    def model_results(name):
        pred = pd.read_csv(output / f"{name}_test_predictions.csv")
        predictions[name] = pred
        audit.check(f"{name}_same_official_test_order", pred.path.tolist() == test.path.tolist())
        audit.check(f"{name}_same_test_labels", np.array_equal(pred.label.to_numpy(), test.label.to_numpy()))
        audit.check(f"{name}_predictions_in_configured_classes", set(pred.prediction).issubset(classes))
        row = metric_table.loc[metric_table.model.eq(name)].iloc[0]
        actual = scores(pred.label, pred.prediction, classes)
        for key, value in actual.items():
            audit.check(f"{name}_{key}_recomputed", np.isclose(float(row[key]), value, atol=1e-12, rtol=0),
                        dict(saved=float(row[key]), recomputed=value))
        audit.check(f"{name}_test_count", int(row.test_count) == len(test) == len(pred))
        for timing in ("train_seconds", "predict_seconds", "predict_ms_per_image"):
            audit.check(f"{name}_{timing}_finite_nonnegative", np.isfinite(row[timing]) and row[timing] >= 0)
        audit.check(f"{name}_prediction_timing_units", np.isclose(
            row.predict_ms_per_image, row.predict_seconds * 1000 / len(test), atol=1e-10, rtol=1e-10))
        cm = pd.read_csv(output / f"{name}_confusion.csv", index_col=0)
        audit.check(f"{name}_confusion_class_order", list(map(int, cm.index)) == classes
                    and list(map(int, cm.columns)) == classes)
        expected_cm = confusion_matrix(pred.label, pred.prediction, labels=classes)
        audit.check(f"{name}_confusion_recomputed", np.array_equal(cm.to_numpy(), expected_cm))
        bundle = joblib.load(output / "models" / f"{name}.joblib")
        audit.check(f"{name}_saved_config_matches_run", bundle["config"] == config)
        audit.check(f"{name}_saved_classes_match_run", list(bundle["classes"]) == classes)
        audit.check(f"{name}_pipeline_classes_match_run", list(bundle["pipeline"].classes_) == classes)
        audit.check(f"{name}_refit_excludes_test_and_quarantined_tracks",
                    bundle["refit_samples"] == len(fit_pool))
        audit.check(f"{name}_training_count_matches_active_manifest",
                    bundle["train_samples"] == int(manifest.split.eq("train").sum()))
        audit.check(f"{name}_validation_count_matches_active_manifest",
                    bundle["val_samples"] == int(manifest.split.eq("val").sum()))

    for name in MODELS:
        audit.phase(f"{name}_results_validation", lambda name=name: model_results(name))

    def recorded_verification():
        recorded = json.loads((output / "verification.json").read_text(encoding="utf-8"))
        audit.check("model_reload_verification_covers_three_models", set(recorded) == set(MODELS))
        for name in MODELS:
            for key in ("reload_batch_predictions_identical", "single_image_matches_batch"):
                audit.check(f"{name}_{key}", recorded[name].get(key) is True)

    audit.phase("recorded_reload_checks", recorded_verification)

    def robustness():
        subset = pd.read_csv(output / "robustness_subset.csv")
        table = pd.read_csv(output / "robustness.csv")
        pred = pd.read_csv(output / "robustness_predictions.csv")
        audit.check("robustness_subset_unique_paths", not subset.path.duplicated().any())
        audit.check("robustness_subset_is_test", set(subset.path).issubset(set(test.path)))
        audit.check("robustness_subset_class_coverage", set(subset.label) == set(classes))
        expected_subset = pd.concat([
            part.sample(min(len(part), config["robust_per_class"]), random_state=config["seed"])
            for _, part in test.groupby("label")]).sort_values(["label", "path"])
        audit.check("robustness_stratified_seeded_sampling", subset.path.tolist() == expected_subset.path.tolist())
        label_lookup = test.set_index("path").label.to_dict()
        audit.check("robustness_subset_labels_match_test", all(
            label_lookup.get(path) == label for path, label in zip(subset.path, subset.label)))
        levels = {"brightness": [1, .75, .5, .25, .1], "blur": [0, .5, 1, 1.5, 2]}
        audit.check("robustness_ten_conditions", len(table) == 10 and set(table.interference) == set(levels))
        baseline_predictions = {}
        for interference, values in levels.items():
            rows = table.loc[table.interference.eq(interference)]
            audit.check(f"{interference}_expected_levels", sorted(rows.strength) == sorted(values))
            for value in values:
                summary = rows.loc[np.isclose(rows.strength, value)]
                group = pred.loc[pred.interference.eq(interference) & np.isclose(pred.strength, value)]
                key = f"robust_{interference}_{value:g}"
                audit.check(f"{key}_one_summary", len(summary) == 1)
                audit.check(f"{key}_same_subset_order", group.path.tolist() == subset.path.tolist())
                audit.check(f"{key}_same_labels", group.label.tolist() == subset.label.tolist())
                audit.check(f"{key}_sample_count", len(group) == len(subset) == int(summary.iloc[0].n))
                for metric, actual in scores(group.label, group.prediction, classes).items():
                    audit.check(f"{key}_{metric}_recomputed", np.isclose(
                        summary.iloc[0][metric], actual, atol=1e-12, rtol=0))
                if (interference == "brightness" and value == 1) or (interference == "blur" and value == 0):
                    baseline_predictions[interference] = group.prediction.to_numpy()
        audit.check("brightness_and_blur_undistorted_baselines_match", np.array_equal(
            baseline_predictions["brightness"], baseline_predictions["blur"]))
        official_lookup = predictions["hog_svm"].set_index("path").prediction
        audit.check("robustness_baseline_matches_fixed_official_model", np.array_equal(
            baseline_predictions["brightness"], official_lookup.loc[subset.path].to_numpy()))

    audit.phase("robustness_validation", robustness)

    def images():
        assets = Path(config["assets_dir"])
        if not assets.is_absolute():
            assets = ROOT / assets
        files = sorted(set(assets.rglob("*.png")) | set((output / "demo_inputs").glob("*.png")))
        audit.check("presentation_png_assets_exist", len(files) > 0, len(files))
        invalid = []
        for path in files:
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.width <= 0 or image.height <= 0:
                        invalid.append(str(path))
            except Exception as exc:
                invalid.append(f"{path}: {exc}")
        audit.check("all_delivery_pngs_readable", not invalid, dict(count=len(files), invalid=invalid))
        singles = list((assets / "04_实验分析" / "perturbation_singles").glob("*.png"))
        audit.check("ten_distinct_perturbation_single_pngs", len(singles) >= 10,
                    [p.name for p in singles])
        for name in ("brightness_0.75", "brightness_0.5", "brightness_0.25", "brightness_0.1",
                     "blur_0.5", "blur_1.5"):
            audit.check(f"decimal_filename_preserved_{name}", any(p.stem == name for p in singles))

    audit.phase("png_delivery_validation", images)

    def source_provenance():
        env = json.loads((output / "environment.json").read_text(encoding="utf-8"))
        recorded = env.get("source_sha256", {})
        changed = []
        for filename, expected in recorded.items():
            path = Path(filename)
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(filename)
        audit.check("source_hashes_recorded", bool(recorded))
        audit.notes.append(dict(source_files_different_since_run=changed,
                                interpretation="Changes require review; source mismatch is recorded, not silently ignored."))

    audit.phase("source_provenance", source_provenance)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/full")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        audit = verify(output)
    except Exception as exc:
        audit = Audit()
        audit.check("required_experiment_inputs", False, f"{type(exc).__name__}: {exc}")
    failed = [item for item in audit.checks if not item["passed"]]
    result = dict(passed=not failed, output=str(output),
                  verified_utc=datetime.now(timezone.utc).isoformat(),
                  check_count=len(audit.checks), failure_count=len(failed),
                  checks=audit.checks, notes=audit.notes)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "delivery_verification.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Delivery verification: {len(audit.checks)} checks, {len(failed)} failures; {target}")
    if failed:
        for item in failed:
            print(f"FAIL {item['check']}: {item['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
