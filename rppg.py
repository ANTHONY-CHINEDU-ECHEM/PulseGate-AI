"""Remote photoplethysmography: a heartbeat read from skin colour.

Blood volume in the skin changes with every heartbeat and modulates the light
reflected from the face by a fraction of a percent. Averaging thousands of skin
pixels per frame makes that modulation measurable with an ordinary camera.
A printed photograph has no pulse, and a replayed video rarely carries one
through compression, the display and a second camera.

Method: plane orthogonal to skin (POS) by Wang, den Brinker, Stuijk and de Haan
(2017), followed by a band pass filter and a spectral peak search. The signal
to noise ratio follows de Haan and Jeanne (2013).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import signal as sps

from ..vision.geometry import FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK


@dataclass
class PulseResult:
    valid: bool
    bpm: float = 0.0
    snr_db: float = -99.0
    seconds: float = 0.0
    score: float = 0.0              # 0 no pulse evidence, 1 strong pulse
    ambient_flicker: bool = False   # the same rhythm is present in the background
    reason: str = ""
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    power: np.ndarray = field(default_factory=lambda: np.zeros(0))
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sample_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "valid": self.valid, "bpm": round(self.bpm, 1), "snr_db": round(self.snr_db, 2),
            "seconds": round(self.seconds, 1), "score": round(self.score, 3),
            "ambient_flicker": self.ambient_flicker, "reason": self.reason,
        }


def skin_mean(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    """Mean colour of forehead and cheeks. Returns ``None`` when the regions leave the frame."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in (FOREHEAD, RIGHT_CHEEK, LEFT_CHEEK):
        pts = np.round(landmarks[list(region), :2]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    x0, y0 = np.maximum(landmarks[:, :2].min(axis=0).astype(int), 0)
    x1, y1 = np.minimum(landmarks[:, :2].max(axis=0).astype(int) + 1, [w, h])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sub_mask = mask[y0:y1, x0:x1]
    if int(sub_mask.sum()) < 50:
        return None
    return np.array(cv2.mean(frame_bgr[y0:y1, x0:x1], mask=sub_mask)[:3], dtype=np.float64)


def background_mean(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Mean colour of the frame outside an enlarged face box, as an ambient light reference."""
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (64, 48), interpolation=cv2.INTER_AREA)
    lo = landmarks[:, :2].min(axis=0)
    hi = landmarks[:, :2].max(axis=0)
    centre, half = (lo + hi) / 2.0, (hi - lo) * 0.9
    mask = np.ones((48, 64), dtype=np.uint8)
    x0, y0 = int(max(0, (centre[0] - half[0]) / w * 64)), int(max(0, (centre[1] - half[1]) / h * 48))
    x1, y1 = int(min(64, (centre[0] + half[0]) / w * 64)), int(min(48, (centre[1] + 2.2 * half[1]) / h * 48))
    mask[y0:y1, x0:x1] = 0
    if int(mask.sum()) < 40:
        return np.array(cv2.mean(small)[:3], dtype=np.float64)
    return np.array(cv2.mean(small, mask=mask)[:3], dtype=np.float64)


def pos_pulse(rgb: np.ndarray, fs: float, window_s: float = 1.6) -> np.ndarray:
    """Plane orthogonal to skin projection with overlap add. ``rgb`` is N x 3 in R, G, B order."""
    n = len(rgb)
    win = max(8, int(round(window_s * fs)))
    out = np.zeros(n)
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for start in range(0, n - win + 1):
        block = rgb[start:start + win]
        normalised = block / (block.mean(axis=0, keepdims=True) + 1e-9)
        s = normalised @ projection.T
        h = s[:, 0] + (s[:, 0].std() / (s[:, 1].std() + 1e-9)) * s[:, 1]
        out[start:start + win] += h - h.mean()
    return out


def chrom_pulse(rgb: np.ndarray) -> np.ndarray:
    """Chrominance method of de Haan and Jeanne, kept as an alternative to POS."""
    normalised = rgb / (rgb.mean(axis=0, keepdims=True) + 1e-9)
    x = 3.0 * normalised[:, 0] - 2.0 * normalised[:, 1]
    y = 1.5 * normalised[:, 0] + normalised[:, 1] - 1.5 * normalised[:, 2]
    return x - (x.std() / (y.std() + 1e-9)) * y


def spectrum(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Zero padded periodogram restricted to a slightly widened band."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(max(n * 8, 256))))
    window = np.hanning(n)
    power = np.abs(np.fft.rfft((x - x.mean()) * window, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = (freqs >= band[0]) & (freqs <= min(band[1] * 2.0, fs / 2.0 * 0.98))
    return freqs[keep], power[keep]


def pulse_snr(freqs: np.ndarray, power: np.ndarray, band: tuple[float, float], peak_hz: float, half_width: float = 0.12) -> float:
    """Power around the peak and its first harmonic over the remaining power, in decibel."""
    inside = (np.abs(freqs - peak_hz) <= half_width) | (np.abs(freqs - 2.0 * peak_hz) <= half_width)
    considered = freqs <= max(band[1], 2.0 * peak_hz + half_width)
    signal_power = power[inside & considered].sum()
    noise_power = power[~inside & considered].sum()
    return float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


class PulseEstimator:
    """Accumulates skin colour over a session and estimates heart rate and its quality."""

    def __init__(self, method: str = "pos", min_seconds: float = 8.0, band_hz: tuple[float, float] = (0.75, 3.0),
                 resample_hz: float = 30.0, snr_live_db: float = 2.0, snr_strong_db: float = 5.0, max_seconds: float = 30.0):
        self.method = method
        self.min_seconds = float(min_seconds)
        self.band = (float(band_hz[0]), float(band_hz[1]))
        self.resample_hz = float(resample_hz)
        self.snr_live_db = float(snr_live_db)
        self.snr_strong_db = float(snr_strong_db)
        self.max_seconds = float(max_seconds)
        self.reset()

    @classmethod
    def from_config(cls, cfg) -> "PulseEstimator":
        r = cfg.rppg
        return cls(r.method, r.min_seconds, tuple(r.band_hz), r.resample_hz, r.snr_live_db, r.snr_strong_db)

    def reset(self) -> None:
        self._t: deque[float] = deque()
        self._skin: deque[np.ndarray] = deque()
        self._ambient: deque[np.ndarray] = deque()

    @property
    def seconds(self) -> float:
        return float(self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0

    def add(self, t: float, frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
        colour = skin_mean(frame_bgr, landmarks)
        if colour is None:
            return
        self.add_sample(t, colour, background_mean(frame_bgr, landmarks))

    def add_sample(self, t: float, skin_bgr: np.ndarray, ambient_bgr: np.ndarray | None = None) -> None:
        self._t.append(float(t))
        self._skin.append(np.asarray(skin_bgr, dtype=np.float64))
        self._ambient.append(np.asarray(skin_bgr if ambient_bgr is None else ambient_bgr, dtype=np.float64))
        while self._t and self._t[-1] - self._t[0] > self.max_seconds:
            self._t.popleft()
            self._skin.popleft()
            self._ambient.popleft()

    def _uniform(self) -> tuple[np.ndarray, np.ndarray, float]:
        t = np.asarray(self._t)
        skin = np.asarray(self._skin)[:, ::-1]        # BGR -> RGB
        ambient = np.asarray(self._ambient)[:, ::-1]
        native = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
        fs = min(self.resample_hz, max(native, 1.0))
        grid = np.arange(t[0], t[-1], 1.0 / fs)
        skin_u = np.column_stack([np.interp(grid, t, skin[:, c]) for c in range(3)])
        ambient_u = np.column_stack([np.interp(grid, t, ambient[:, c]) for c in range(3)])
        return skin_u, ambient_u, fs

    def _extract(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        raw = chrom_pulse(rgb) if self.method == "chrom" else pos_pulse(rgb, fs)
        high = min(self.band[1], fs / 2.0 * 0.9)
        sos = sps.butter(3, [self.band[0], high], btype="bandpass", fs=fs, output="sos")
        return sps.sosfiltfilt(sos, raw)

    def estimate(self) -> PulseResult:
        if self.seconds < self.min_seconds or len(self._t) < 40:
            return PulseResult(False, seconds=self.seconds, reason="not enough video")
        skin, ambient, fs = self._uniform()
        if fs < 2.5 * self.band[1]:
            return PulseResult(False, seconds=self.seconds, reason="frame rate too low")
        wave = self._extract(skin, fs)
        freqs, power = spectrum(wave, fs, self.band)
        in_band = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if not in_band.any() or power[in_band].max() <= 0:
            return PulseResult(False, seconds=self.seconds, reason="flat signal")
        peak_hz = float(freqs[in_band][np.argmax(power[in_band])])
        snr = pulse_snr(freqs, power, self.band, peak_hz)

        # Ambient check: lamps and displays flicker, and that rhythm shows up everywhere in the frame.
        flicker = False
        green = ambient[:, 1] / (ambient[:, 1].mean() + 1e-9)
        skin_green = skin[:, 1] / (skin[:, 1].mean() + 1e-9)
        if green.std() > 1e-6:
            sos = sps.butter(3, [self.band[0], min(self.band[1], fs / 2.0 * 0.9)], btype="bandpass", fs=fs, output="sos")
            a = sps.sosfiltfilt(sos, green)
            s = sps.sosfiltfilt(sos, skin_green)
            fa, pa = spectrum(a, fs, self.band)
            _, ps = spectrum(s, fs, self.band)
            near = np.abs(fa - peak_hz) <= 0.12
            if near.any() and ps[near].sum() > 0:
                flicker = bool(pa[near].sum() >= 0.5 * ps[near].sum() and pulse_snr(fa, pa, self.band, peak_hz) > 0.0)

        span = max(self.snr_strong_db - self.snr_live_db, 1e-6)
        score = float(np.clip((snr - self.snr_live_db) / span * 0.5 + 0.5, 0.0, 1.0)) if snr >= self.snr_live_db - span else 0.0
        if flicker:
            score = 0.0
        return PulseResult(
            True, bpm=peak_hz * 60.0, snr_db=snr, seconds=self.seconds, score=score, ambient_flicker=flicker,
            reason="ambient flicker" if flicker else "", freqs=freqs, power=power, waveform=wave, sample_rate=fs,
        )
