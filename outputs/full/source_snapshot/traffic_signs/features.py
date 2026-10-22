"""Shared, deterministic image preprocessing and cached feature extraction.

Distortions are applied at source resolution after optional ROI cropping, in the
order brightness then Gaussian blur. Blur radius is measured in source pixels.
Pillow RGB -> L uses ITU-R 601 luminance; the resulting values are divided by 255.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import skimage
from skimage.feature import hog

PREPROCESS_VERSION = 1
# At 32x32 float32, this is bounded to about 240 MB. Only audited, immutable
# manifest rows use this per-process cache; external inputs are read normally.
_GRAY_CACHE = OrderedDict()
_GRAY_CACHE_LIMIT = 60000

def _audited_hash(row):
    value=getattr(row,'sha256',None)
    return value if isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdefABCDEF' for c in value) else None


def _size(config: dict) -> tuple[int, int]:
    size = config.get("image_size", 32)
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2 or min(size) < 1:
        raise ValueError("image_size must be positive integer or [width, height]")
    return tuple(map(int, size))


def _hog_options(config: dict) -> dict:
    options = {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "transform_sqrt": True,
    }
    options.update(config.get("hog", {}))
    options["pixels_per_cell"] = tuple(options["pixels_per_cell"])
    options["cells_per_block"] = tuple(options["cells_per_block"])
    return options


def load_rgb(path, config: dict, roi=None, brightness=1.0, blur=0.0) -> Image.Image:
    """Load and resize RGB with the identical preprocessing used for inference.

    ROI coordinates are inclusive as in the official GTSRB annotations. A model
    trained with ROI cropping requires explicit coordinates at prediction time.
    """
    if not np.isfinite(brightness) or brightness < 0:
        raise ValueError("brightness must be finite and nonnegative")
    if not np.isfinite(blur) or blur < 0:
        raise ValueError("blur must be finite and nonnegative")
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    if config.get("use_roi", False):
        if roi is None or len(roi) != 4 or not np.all(np.isfinite(roi)):
            raise ValueError("ROI preprocessing requires four explicit coordinates")
        x1, y1, x2, y2 = map(int, roi)
        if not (0 <= x1 <= x2 < rgb.width and 0 <= y1 <= y2 < rgb.height):
            raise ValueError(f"ROI {roi} is outside image {path}: {rgb.size}")
        rgb = rgb.crop((x1, y1, x2 + 1, y2 + 1))
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(float(brightness))
    if blur:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    return rgb.resize(_size(config), resample=Image.Resampling.BILINEAR)


def preprocess(path, config: dict, roi=None, brightness=1.0, blur=0.0) -> np.ndarray:
    """Return float32 grayscale image in [0, 1]. No fitted transformation."""
    rgb = load_rgb(path, config, roi=roi, brightness=brightness, blur=blur)
    return np.asarray(rgb.convert("L"), dtype=np.float32) / np.float32(255.0)


def feature_vector(gray: np.ndarray, config: dict, kind="hog") -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or not np.all(np.isfinite(gray)):
        raise ValueError("Expected finite two-dimensional grayscale image")
    if kind == "pixels":
        return gray.ravel().copy()
    if kind == "hog":
        return np.asarray(hog(gray, **_hog_options(config), feature_vector=True), dtype=np.float32)
    raise ValueError(f"Unknown feature kind: {kind!r}")


def hog_visualization(gray: np.ndarray, config: dict) -> np.ndarray:
    """Return HOG orientation rendering; display with contrast rescaling only."""
    _, visual = hog(np.asarray(gray, dtype=np.float32), **_hog_options(config),
                    visualize=True, feature_vector=True)
    return np.asarray(visual, dtype=np.float32)


def _cache_identity(manifest, config, kind, brightness, blur):
    records = []
    use_roi = config.get("use_roi", False)
    for row in manifest.itertuples(index=False):
        # Manifests already contain absolute checked paths. Avoid resolving every
        # ancestor on Windows for tens of thousands of images at each cache call.
        path = Path(os.path.abspath(row.path))
        record: dict[str, Any] = {
            "path": str(path),
        }
        if _audited_hash(row) is None:
            stat = path.stat()
            record.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        for field in ("label", "split", "track", "sha256"):
            if hasattr(row, field):
                record[field] = str(getattr(row, field))
        if use_roi:
            record["roi"] = [int(getattr(row, key)) for key in ("x1", "y1", "x2", "y2")]
        records.append(record)
    identity = {
        "preprocess_version": PREPROCESS_VERSION,
        "numpy": np.__version__, "pillow": PIL.__version__, "skimage": skimage.__version__,
        "kind": kind, "image_size": _size(config), "use_roi": bool(use_roi),
        "grayscale": "Pillow-L-ITU-R-601/255", "resize": "Pillow-BILINEAR",
        "distortion_order": "roi-brightness-blur-resize-gray", "brightness": float(brightness),
        "blur_source_pixel_radius": float(blur), "hog": _hog_options(config),
        "records": records,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), identity


def extract_features(manifest, config: dict, kind="hog", brightness=1.0, blur=0.0,
                     cache=True) -> np.ndarray:
    """Return rows aligned exactly to manifest order, with optional .npy cache.

    Cache identity includes file paths, either audited content checksums or
    size/nanosecond mtime, plus ordering and all preprocessing parameters.
    External manifests without valid checksums fall back to file metadata.
    """
    if kind not in ("hog", "pixels"):
        raise ValueError(f"Unknown feature kind: {kind!r}")
    cache_file = None
    if cache:
        key, identity = _cache_identity(manifest, config, kind, brightness, blur)
        cache_dir = Path(config.get("output_dir", "outputs/full")) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{kind}_{key}.npy"
        if cache_file.exists():
            result = np.load(cache_file, allow_pickle=False)
            if result.ndim == 2 and result.shape[0] == len(manifest) and result.dtype == np.float32:
                return result
            raise ValueError(f"Invalid feature cache; remove {cache_file}")
    width, height = _size(config)
    dimension = len(feature_vector(np.zeros((height, width), np.float32), config, kind))
    result = np.empty((len(manifest), dimension), dtype=np.float32)
    for i, row in enumerate(manifest.itertuples(index=False)):
        roi = ([getattr(row, key) for key in ("x1", "y1", "x2", "y2")]
               if config.get("use_roi", False) else None)
        gray_key = None
        if _audited_hash(row) is not None:
            gray_key=(str(row.path),str(row.sha256),_size(config),bool(config.get('use_roi',False)),
                      tuple(roi) if roi else None,float(brightness),float(blur),PREPROCESS_VERSION)
        if gray_key is not None and gray_key in _GRAY_CACHE:
            gray=_GRAY_CACHE.pop(gray_key)
            _GRAY_CACHE[gray_key]=gray
        else:
            gray=preprocess(row.path, config, roi, brightness, blur)
            if gray_key is not None:
                _GRAY_CACHE[gray_key]=gray
                if len(_GRAY_CACHE)>_GRAY_CACHE_LIMIT:
                    _GRAY_CACHE.popitem(last=False)
        result[i] = feature_vector(gray, config, kind)
    if cache_file is not None:
        temporary = cache_file.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(temporary, result, allow_pickle=False)
        temporary.replace(cache_file)
        # Store a compact audit record; the manifest already stores image records.
        audit = {k: v for k, v in identity.items() if k != "records"}
        audit.update({"manifest_rows": len(manifest), "shape": list(result.shape), "sha256": key})
        cache_file.with_suffix(".json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
