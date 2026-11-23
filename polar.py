"""Polar unwrapping of ring-shaped inspection images.

A ring (annulus) image is resampled on a (radius x angle) grid so that the ring becomes a
rectangular strip. Standard CNN encoders and grid patching then see the ring structure with
a consistent orientation, and the empty centre / background outside the ring is dropped.

Pipeline (`unwrap_ring`):
  1. Threshold the image and fit the minimum enclosing circle of the largest contour
     -> ring centre (cx, cy) and an initial outer radius.
  2. Coarse polar resampling over [0, r_max]; the radial mean profile locates the band of
     radii that actually carries signal (mean > signal_ratio * overall mean).
  3. Final polar resampling restricted to that band -> (r_bins, s_bins) strip.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def ring_center(img: np.ndarray, threshold: float = 0.0) -> Tuple[float, float, float]:
    """Centre (x, y) and radius of the minimum enclosing circle of the brightest connected region."""
    h, w = img.shape
    max_val = float(img.max())
    if max_val > 0:
        norm = (img / max_val * 255).astype(np.uint8)
        thr = int((threshold / max_val) * 255)
    else:
        norm, thr = img.astype(np.uint8), 0
    _, binary = cv2.threshold(norm, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (w - 1) / 2.0, (h - 1) / 2.0, min(w, h) / 2.0
    (x, y), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return float(x), float(y), float(radius)


def _bilinear(image: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x0, y0 = np.floor(xs).astype(np.int32), np.floor(ys).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)
    out = wa * image[y0c, x0c] + wb * image[y1c, x0c] + wc * image[y0c, x1c] + wd * image[y1c, x1c]
    return out.astype(np.float32)


def polar_resample(img: np.ndarray, cx: float, cy: float, r_bins: int, s_bins: int,
                   rmin: float = 0.0, rmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sample `img` on r_bins radii x s_bins angles around (cx, cy). Returns (strip, radii)."""
    h, w = img.shape
    if rmax is None:
        rmax = float(min(cx, cy, w - 1 - cx, h - 1 - cy))
    rs = np.linspace(rmin, rmax, r_bins, dtype=np.float32)
    th = np.linspace(0.0, 2 * math.pi, s_bins, endpoint=False, dtype=np.float32)
    R, T = np.meshgrid(rs, th, indexing="ij")
    return _bilinear(img, cy + R * np.sin(T), cx + R * np.cos(T)), rs


def unwrap_ring(img: np.ndarray, r_bins: int = 512, s_bins: int = 1024, threshold: float = 0.0,
                signal_ratio: float = 0.2) -> np.ndarray:
    """Ring image (H, W) -> polar strip (r_bins, s_bins) focused on the signal-carrying radii."""
    cx, cy, r0 = ring_center(img, threshold)
    coarse, rs = polar_resample(img, cx, cy, r_bins, s_bins, 0.0, r0)
    profile = coarse.mean(axis=1)
    idx = np.where(profile > float(profile.mean()) * signal_ratio)[0]
    rmin, rmax = (float(rs[idx.min()]), float(rs[idx.max()])) if len(idx) else (0.0, float(r0))
    strip, _ = polar_resample(img, cx, cy, r_bins, s_bins, rmin, rmax)
    return strip


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Min-max normalise a single-channel image and map it to RGB with the JET colormap."""
    lo, hi = float(img.min()), float(img.max())
    norm = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img, dtype=np.float32)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def grid_patches(img_rgb: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """Non-padded sliding-window patches, (N, patch, patch, 3)."""
    H, W, _ = img_rgb.shape
    out = [img_rgb[i:i + patch, j:j + patch] for i in range(0, H - patch + 1, stride)
           for j in range(0, W - patch + 1, stride)]
    return np.stack(out) if out else np.zeros((0, patch, patch, 3), dtype=img_rgb.dtype)
