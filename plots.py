"""Figures for the reports and the documentation.

Every figure is rebuilt from the JSON and CSV files in ``reports``, so the
pictures can never drift away from the numbers. Faces in figures come only from
the openly licensed sample footage, never from the training database.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..config import PROJECT_ROOT, Config
from .style import (
    ATTACK, CLASS_COLOURS, CLASS_LABELS, GRID, INK, INK_MUTED, INK_SOFT, SEQUENTIAL, SERIES, SURFACE, apply_style, save,
)

SPECIES_ORDER = ["print_matte", "print_glossy", "replay_phone", "replay_monitor", "cutout_mask"]


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _pct(ax, axis: str = "y", decimals: int = 0) -> None:
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Diagrams and picture examples
# ----------------------------------------------------------------------------

def figure_architecture(out: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 70)
    ax.axis("off")

    def box(x, y, w, h, title, lines, colour, fill="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.4", linewidth=1.4, edgecolor=colour, facecolor=fill))
        ax.add_patch(FancyBboxPatch((x, y + h - 1.1), w, 1.1, boxstyle="round,pad=0.2,rounding_size=1.0", linewidth=0, facecolor=colour))
        ax.text(x + 1.6, y + h - 3.9, title, fontsize=10.5, fontweight="bold", color=INK, va="center")
        for i, line in enumerate(lines):
            ax.text(x + 1.6, y + h - 7.3 - 2.75 * i, line, fontsize=8.8, color=INK_SOFT, va="center")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=INK_MUTED, shrinkA=0, shrinkB=0))

    blue, orange, aqua, yellow, violet = SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[6]
    box(1, 24.5, 20, 14, "Camera", ["webcam or phone", "live or recorded"], INK_MUTED)
    box(27, 21.5, 24, 20, "Face tracking", ["YuNet face detector", "468 point 3D face mesh", "head pose, eyes, mouth", "capture quality gate"], blue)
    signals = [
        (48.5, "Passive texture model", ["PulseGateNet, 0.96 M weights", "context view and native patch"], orange),
        (35.5, "Random challenges", ["blink, turn, chin, mouth", "random order and timing"], aqua),
        (22.5, "Depth from motion", ["is the face a rigid 3D shape", "or a tilted flat surface"], yellow),
        (9.5, "Remote pulse", ["heartbeat from skin colour", "supporting evidence only"], violet),
    ]
    for y, title, lines, colour in signals:
        box(58, y, 30, 11.5, title, lines, colour)
        arrow(51.6, 31.5, 57.4, y + 5.75)
        arrow(88.6, y + 5.75, 95.4, 31.5)
    box(96, 21.5, 28, 20, "Fusion and policy", ["hard gates per signal", "weighted confidence score", "live, not live or retry", "JSON record for audit"], INK)
    arrow(21.6, 31.5, 26.4, 31.5)
    ax.text(1, 67.5, "PulseGate AI: four independent signals, one decision", fontsize=13, fontweight="bold", color=INK)
    ax.text(1, 63.8, "Each signal defeats a different attack, so an impostor has to beat all of them in the same session.", fontsize=9.8, color=INK_SOFT)
    ax.text(1, 3.6, "A photograph fails the challenges.   A recording fails random prompts.   A tilted print fails the depth test.   A screen relaying a live accomplice fails the texture model.",
            fontsize=8.6, color=INK_MUTED, va="center")
    ax.plot([1, 124], [7.0, 7.0], color=GRID, linewidth=1.0)
    return save(fig, out)


def render_examples(cfg: Config, seed: int = 4) -> dict[str, np.ndarray]:
    """One genuine frame and one frame per attack species, all from the sample footage."""
    from ..data.attacks.common import face_ref_from_detection
    from ..data.attacks.simulator import LIVE, SPECIES, Simulator
    from ..vision.tracker import FaceTracker

    tracker = FaceTracker(cfg)
    sample_dir = cfg.path("paths.video_dir")
    source = cv2.imread(str(PROJECT_ROOT / "assets" / "samples" / "face.jpg"))
    background = None
    other = sample_dir / "head_pose_face_detection_male.mp4"
    if other.exists():
        cap = cv2.VideoCapture(str(other))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ok, frame = cap.read()
        cap.release()
        background = frame if ok else None
    if background is None:
        background = np.ascontiguousarray(source[:, ::-1])
    # enlarge so that faces have the size the simulator was designed for
    source = cv2.resize(source, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    background = cv2.resize(background, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    face = face_ref_from_detection(tracker.detect(source)[0])
    bg_face = face_ref_from_detection(tracker.detect(background)[0])
    sim = Simulator((480, 640))
    out: dict[str, np.ndarray] = {}
    for i, species in enumerate((LIVE,) + SPECIES):
        rng = np.random.default_rng(seed * 100 + i)
        params = sim.sample(species, rng)
        # a clean, fixed camera so that the differences between the media are easy to see
        cam = params.camera
        cam.blur_sigma, cam.motion_length, cam.downscale, cam.jpeg_quality = 0.3, 0, 1.0, 92
        cam.shot_noise, cam.read_noise, cam.sharpen, cam.saturation, cam.flare = 0.0008, 0.003, 0.0, 1.0, 0.0
        cam.gain, cam.white_balance, cam.exposure_target, cam.gamma, cam.contrast = 1.0, (1.0, 1.0, 1.0), 0.2, 2.2, 1.0
        medium = getattr(params.medium, "sheet", params.medium)
        if medium is not None:
            medium.face_width = 190.0
        if params.live is not None:
            params.live.face_width = 190.0
        if species == "replay_monitor":
            params.medium.pitch = 0.75
        if species == "replay_phone":
            params.medium.content_width = 1.7
            params.medium.bezel = 0.05
        if species == "print_matte":
            params.medium.process, params.medium.cell, params.medium.border, params.medium.content_width = "halftone", 1.3, 0.05, 1.7
        if species == "print_glossy":
            params.medium.content_width, params.medium.border = 1.8, 0.03
        if species == "cutout_mask":
            params.background.face_width = 190.0
            params.medium.sheet.process, params.medium.eye_holes, params.medium.rim = "stochastic", 0.25, 1.5
        if params.background is not None:
            params.background.gain, params.background.gamma, params.background.light = 1.0, 1.0, (0.0, 0.0)
        if params.live is not None:
            params.live.gain, params.live.gamma, params.live.light, params.live.flip = 1.0, 1.0, (0.0, 0.0), False
        frame, _ = sim.render(source, face, params, rng, background, bg_face)
        out[species] = frame
    return out


def figure_attack_gallery(cfg: Config, out: Path, model=None) -> Path:
    """What the camera sees for a real face and for each simulated attack, with a magnified skin patch."""
    from ..vision.geometry import crop_square, square_box
    from ..vision.tracker import FaceTracker

    apply_style()
    frames = render_examples(cfg)
    tracker = FaceTracker(cfg)
    names = ["live"] + SPECIES_ORDER
    fig, axes = plt.subplots(2, 6, figsize=(14.5, 6.55), gridspec_kw={"height_ratios": [4, 3], "hspace": 0.03, "wspace": 0.05})
    for column, name in enumerate(names):
        frame = frames[name]
        obs = tracker.process_image(frame)
        top, bottom = axes[0, column], axes[1, column]
        top.imshow(_rgb(frame))
        title = CLASS_LABELS[name]
        if obs is not None:
            cx, cy, side = square_box(obs.box, cfg.crop.context_scale)
            crop = crop_square(frame, cx, cy, side)
            c = crop.shape[0] // 2
            patch = crop[c - 40:c + 40, c - 40:c + 40]
            bottom.imshow(_rgb(cv2.resize(patch, (320, 320), interpolation=cv2.INTER_NEAREST)))
            if model is not None:
                score = model.score_crop(crop)
                title += f"\nlive score {score.live:.2f}"
        top.set_title(title, fontsize=10.5, loc="center", pad=6)
        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(CLASS_COLOURS[name])
                spine.set_linewidth(2.2)
    axes[1, 0].set_ylabel("skin patch, 4x", fontsize=9.5)
    axes[0, 0].set_ylabel("camera frame", fontsize=9.5)
    fig.suptitle("One genuine presentation and five attack instruments, as rendered by the simulator", x=0.125, ha="left", fontsize=12.5, fontweight="bold", y=0.94)
    fig.text(0.125, 0.06, "Source footage: Intel IoT DevKit sample videos, CC BY 4.0. Attack frames are simulations of that footage shown on paper or on a display.",
             fontsize=8.5, color=INK_MUTED)
    return save(fig, out)


def figure_hud_sequence(cfg: Config, out: Path) -> Path | None:
    """Screens of a complete interactive session on real footage."""
    from ..apps.hud import draw_hud
    from ..engine.session import LivenessEngine, Stage

    clip = PROJECT_ROOT / "assets" / "samples" / "live_clip.mp4"
    if not clip.exists() or not cfg.path("assets.passive_model").exists():
        return None
    apply_style()
    engine = LivenessEngine(cfg)
    session = engine.new_session("interactive", seed=3, plan=["look_up", "turn_right", "turn_left"])
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wanted = {"positioning": None, "look_up": None, "turn_right": None, "turn_left": None, "hold": None, "done": None}
    index = 0
    running: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        status = session.update(frame, index / fps)
        index += 1
        if status.passive is not None:
            running = (running + [status.passive.live])[-15:]
        estimate = session.pulse.estimate()
        pulse = f"{estimate.bpm:.0f} bpm  {estimate.snr_db:+.1f} dB" if estimate.valid else ""
        key = None
        if status.stage == Stage.POSITIONING and index == 12:
            key = "positioning"
        elif status.stage == Stage.CHALLENGE and status.challenge_progress > 0.75:
            key = session.records[-1].challenge.value
        elif status.stage == Stage.HOLD:
            key = "hold"
        elif status.stage == Stage.DONE:
            key = "done"
        if key and wanted.get(key) is None:
            view = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            scaled = _scale_status(status, 1.5)
            wanted[key] = draw_hud(view, scaled, session.passive_threshold, pulse_text=pulse, fps=fps,
                                   passive_running=float(np.mean(running)) if running else None)
        if status.stage == Stage.DONE:
            break
    cap.release()
    shots = [v for v in wanted.values() if v is not None]
    if not shots:
        return None
    columns = 3
    rows = int(np.ceil(len(shots) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.0 * rows), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, shot in zip(np.ravel(axes), shots):
        ax.imshow(_rgb(shot))
    fig.suptitle("A complete interactive session on real footage: positioning, three prompts, verdict", x=0.125, ha="left", fontsize=12.5, fontweight="bold", y=0.955)
    fig.text(0.125, 0.065, "The prompts in this run were chosen to match what the person in the clip does. In live use they are random. Footage: Intel IoT DevKit sample videos, CC BY 4.0.",
             fontsize=8.5, color=INK_MUTED)
    return save(fig, out)


def _scale_status(status, factor: float):
    """Copy of a status whose face geometry is scaled for drawing on an enlarged frame."""
    import copy

    scaled = copy.copy(status)
    if status.observation is not None:
        obs = copy.copy(status.observation)
        obs.landmarks = status.observation.landmarks * factor
        obs.box = status.observation.box * factor
        scaled.observation = obs
    return scaled


def figure_gradcam(cfg: Config, out: Path) -> Path | None:
    from ..models.gradcam import GradCam, load_checkpoint, overlay
    from ..vision.geometry import crop_square, square_box
    from ..vision.tracker import FaceTracker

    checkpoint = cfg.path("paths.models_dir") / f"{cfg.train.run_name}.pt"
    if not checkpoint.exists():
        return None
    apply_style()
    cam = GradCam(load_checkpoint(checkpoint))
    tracker = FaceTracker(cfg)
    frames = render_examples(cfg, seed=9)
    names = ["live"] + SPECIES_ORDER
    fig, axes = plt.subplots(2, 6, figsize=(14.5, 5.3), gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    results = {}
    for name in names:
        obs = tracker.process_image(frames[name])
        if obs is None:
            continue
        cx, cy, side = square_box(obs.box, cfg.crop.context_scale)
        crop = crop_square(frames[name], cx, cy, side)
        heat, live = cam.explain(crop, cfg.crop.context_scale, cfg.crop.context_size, cfg.crop.patch_size, towards="attack", normalise=False)
        results[name] = (crop, heat, live)
    # one colour scale for the whole row, so that weak evidence looks weak
    peak = max(float(np.percentile(heat, 99.5)) for _, heat, _ in results.values()) + 1e-8
    for column, name in enumerate(names):
        if name not in results:
            continue
        crop, heat, live = results[name]
        heat = np.clip(heat / peak, 0.0, 1.0)
        axes[0, column].imshow(_rgb(crop))
        axes[1, column].imshow(_rgb(overlay(crop, heat, 0.75)))
        axes[0, column].set_title(f"{CLASS_LABELS[name]}\nlive score {live:.2f}", fontsize=10.5, loc="center", pad=6)
        for ax in axes[:, column]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_color(CLASS_COLOURS[name])
                spine.set_linewidth(2.2)
                spine.set_visible(True)
    axes[0, 0].set_ylabel("model input", fontsize=9.5)
    axes[1, 0].set_ylabel("evidence for attack", fontsize=9.5)
    fig.suptitle("Where the context stream finds evidence of an attack", x=0.125, ha="left", fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.125, 0.03, "Grad CAM on the context stream, one brightness scale for all six. Bright regions pushed the score towards attack. Footage: Intel IoT DevKit sample videos, CC BY 4.0.",
             fontsize=8.5, color=INK_MUTED)
    return save(fig, out)


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------

def figure_training(meta: dict, out: Path) -> Path:
    """Loss and validation error over all training stages recorded with the model."""
    apply_style()
    stages = [e.get("history", []) for e in meta.get("earlier_stages", [])] + [meta["history"]]
    rows, boundaries, offset = [], [], 0
    for history in stages:
        for row in history:
            rows.append({**row, "epoch": row["epoch"] + offset})
        offset += len(history)
        boundaries.append(offset)
    history = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    a.plot(history["epoch"], history["train_loss"], color=SERIES[0], marker="o", markersize=4)
    a.set_title("Training loss")
    a.set_xlabel("epoch")
    series = [("val_eer", "Both streams", SERIES[0]), ("val_eer_context", "Context stream only", SERIES[1]), ("val_eer_texture", "Texture stream only", SERIES[2])]
    for column, label, colour in series:
        b.plot(history["epoch"], 100 * history[column], color=colour, label=label, marker="o", markersize=4)
        b.annotate(f"{100 * history[column].iloc[-1]:.1f}%", (history["epoch"].iloc[-1], 100 * history[column].iloc[-1]), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=9, color=INK_SOFT)
    b.set_title("Validation equal error rate")
    b.set_xlabel("epoch")
    b.set_ylim(0, None)
    _pct(b)
    b.legend(loc="lower left")
    for ax in (a, b):
        ax.set_xticks(history["epoch"][::2])
        for boundary in boundaries[:-1]:
            ax.axvline(boundary + 0.5, color=INK_MUTED, linewidth=1.0)
            ax.annotate("stage two", (boundary + 0.5, ax.get_ylim()[1]), xytext=(5, -3), textcoords="offset points",
                        ha="left", va="top", fontsize=8.5, color=INK_SOFT)
    return save(fig, out)


def figure_score_distribution(scores: pd.DataFrame, threshold: float, out: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.12})
    bins = np.linspace(0, 1, 51)
    live = scores[scores["label"] == 1]["score"]
    axes[0].hist(live, bins=bins, color=CLASS_COLOURS["live"], edgecolor=SURFACE, linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Model score on the test subjects")
    axes[0].text(0.01, 0.86, f"Live, {len(live):,} presentations", transform=axes[0].transAxes, fontsize=10, color=INK)
    stack = [scores[scores["species"] == s]["score"] for s in SPECIES_ORDER]
    axes[1].hist(stack, bins=bins, stacked=True, color=[CLASS_COLOURS[s] for s in SPECIES_ORDER], label=[CLASS_LABELS[s] for s in SPECIES_ORDER],
                 edgecolor=SURFACE, linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].text(0.99, 0.9, f"Attacks, {int((scores['label'] == 0).sum()):,} presentations", transform=axes[1].transAxes, fontsize=10, color=INK, ha="right")
    axes[1].legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.22))
    axes[1].set_xlabel("score: estimated probability that the presentation is a live person")
    for ax in axes:
        ax.axvline(threshold, color=INK, linewidth=1.2)
        ax.set_ylabel("presentations")
    axes[0].annotate(f"threshold {threshold:.2f}\nfixed on validation data", (threshold, axes[0].get_ylim()[1]), xytext=(-8, -6), textcoords="offset points",
                     ha="right", va="top", fontsize=9, color=INK_SOFT)
    return save(fig, out)


def _det(ax, labels, score, label, colour):
    from .metrics import error_curves

    _, apcer, bpcer = error_curves(labels, score)
    floor = 2e-4
    ax.plot(100 * np.maximum(apcer, floor), 100 * np.maximum(bpcer, floor), color=colour, label=label)


def figure_det(scores: pd.DataFrame, baseline: pd.DataFrame | None, metrics: dict, out: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    curves = [("score", "PulseGateNet, both streams", SERIES[0]), ("score_context", "Context stream only", SERIES[1]), ("score_texture", "Texture stream only", SERIES[2])]
    for column, label, colour in curves:
        _det(ax, scores["label"], scores[column], label, colour)
    if baseline is not None:
        _det(ax, baseline["label"], baseline["score"], "Colour texture baseline", SERIES[3])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 60)
    ax.set_ylim(0.5, 60)
    ticks = [0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [f"{t:g}%" for t in ticks])
    ax.set_yticks(ticks, [f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.plot([0.5, 60], [0.5, 60], color=GRID, linewidth=1.2)
    ax.set_xlabel("attacks accepted (APCER)")
    ax.set_ylabel("genuine presentations rejected (BPCER)")
    ax.set_title("Detection error trade off on the test subjects")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.06))
    ax.text(0.03, 0.02, "lower left is better, the diagonal marks equal error rates", transform=ax.transAxes, fontsize=8.5, color=INK_MUTED)
    return save(fig, out)


def figure_species(metrics: dict, unseen: dict | None, out: Path) -> Path:
    """Attack acceptance per instrument: shipped model, and the unseen attack experiment."""
    apply_style()
    per = metrics["metrics"]["per_species"]
    panels = 2 if unseen and unseen.get("families") else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.4 * panels, 4.0), sharey=True, gridspec_kw={"wspace": 0.06})
    axes = np.atleast_1d(axes)
    y = np.arange(len(SPECIES_ORDER))[::-1]
    shipped = [100 * per[s]["apcer"] for s in SPECIES_ORDER]
    axes[0].barh(y, shipped, height=0.5, color=SERIES[0])
    for yi, v in zip(y, shipped):
        axes[0].text(v, yi, f"  {v:.1f}%", va="center", fontsize=9.5, color=INK)
    axes[0].set_yticks(y, [CLASS_LABELS[s] for s in SPECIES_ORDER])
    axes[0].set_title("Shipped model")
    axes[0].set_xlim(0, max(10.0, 1.3 * max(shipped)))
    top = max(shipped)
    if panels == 2:
        seen = [100 * unseen["reference"]["apcer"][s] for s in SPECIES_ORDER]
        novel = []
        for s in SPECIES_ORDER:
            value = next((f["apcer_unseen"][s] for f in unseen["families"].values() if s in f["apcer_unseen"]), None)
            novel.append(0.0 if value is None else 100 * value)
        axes[1].barh(y + 0.19, seen, height=0.34, color=SERIES[0], label="family seen in training")
        axes[1].barh(y - 0.19, novel, height=0.34, color=SERIES[1], label="family never seen in training")
        for yi, a, b in zip(y, seen, novel):
            axes[1].text(a, yi + 0.19, f"  {a:.1f}%", va="center", fontsize=9, color=INK)
            axes[1].text(b, yi - 0.19, f"  {b:.1f}%", va="center", fontsize=9, color=INK)
        axes[1].set_title("Unseen attack experiment, reduced training budget")
        axes[1].legend(loc="upper right")
        axes[1].set_xlim(0, min(108.0, 1.25 * max(seen + novel)))
        top = max(top, max(seen + novel))
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("attacks accepted at the operating threshold (APCER)")
        _pct(ax, "x")
    return save(fig, out)


def figure_confusion(metrics: dict, out: Path) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    apply_style()
    classes = metrics["confusion"]["classes"]
    matrix = np.array(metrics["confusion"]["matrix"], dtype=float)
    share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.imshow(share, cmap=LinearSegmentedColormap.from_list("blue", SEQUENTIAL), vmin=0, vmax=1)
    labels = [CLASS_LABELS[c] for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if share[i, j] >= 0.005:
                ax.text(j, i, f"{100 * share[i, j]:.0f}%", ha="center", va="center", fontsize=9.5, color="#ffffff" if share[i, j] > 0.55 else INK)
    ax.set_xlabel("named by the model")
    ax.set_ylabel("actual presentation")
    ax.set_title("Which attack does the model think it sees")
    ax.grid(False)
    return save(fig, out)


def figure_robustness(metrics: dict, out: Path) -> Path:
    apply_style()
    data = metrics["robustness"]
    panels = [
        ("jpeg_quality", "JPEG quality", True), ("gaussian_blur_sigma", "Blur, sigma in pixels", False),
        ("brightness_gain", "Brightness gain", False), ("sensor_noise_sigma", "Added noise, grey levels", False),
        ("resolution_factor", "Resolution kept", True),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.5), sharey=True, gridspec_kw={"wspace": 0.08})
    top = 5.0
    for ax, (key, label, reverse) in zip(axes, panels):
        rows = sorted(data[key], key=lambda r: r["level"], reverse=reverse)
        x = np.arange(len(rows))
        ax.plot(x, [100 * r["bpcer"] for r in rows], color=CLASS_COLOURS["live"], marker="o", markersize=5, label="Genuine rejected (BPCER)")
        ax.plot(x, [100 * r["apcer"] for r in rows], color=ATTACK, marker="o", markersize=5, label="Attacks accepted (APCER)")
        ax.set_xticks(x, [f"{r['level']:g}" for r in rows])
        ax.set_xlabel(label)
        top = max(top, max(100 * max(r["bpcer"], r["apcer"]) for r in rows))
    axes[0].set_ylim(-0.02 * top, 1.1 * top)
    _pct(axes[0])
    axes[0].set_title("Error rates when the test images are degraded, threshold unchanged")
    axes[0].legend(loc="upper left")
    fig.text(0.125, -0.06, "Each panel worsens one property from left to right. The model was not retrained or recalibrated.", fontsize=8.5, color=INK_MUTED)
    return save(fig, out)


def figure_slices(metrics: dict, out: Path) -> Path:
    apply_style()
    slices = metrics["slices"]
    panels = [("gender", "Gender"), ("glasses", "Glasses"), ("camera", "Camera view"), ("lighting", "Lighting set"), ("face_size", "Face size")]
    widths = [len(slices[k]) for k, _ in panels]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.6), sharey=True, gridspec_kw={"width_ratios": widths, "wspace": 0.08})
    top = 1.0
    for ax, (key, label) in zip(axes, panels):
        names = list(slices[key])
        values = [100 * (slices[key][n]["bpcer"] or 0) for n in names]
        x = np.arange(len(names))
        ax.bar(x, values, width=min(0.6, 0.12 * len(names) + 0.25), color=CLASS_COLOURS["live"])
        short = [n.replace(" px", "").replace("under ", "<").replace("over ", ">").replace(" to ", " to\n") for n in names]
        ax.set_xticks(x, short, fontsize=8.5)
        ax.set_xlabel(label)
        ax.grid(axis="x", visible=False)
        top = max(top, max(values))
        for xi, v, n in zip(x, values, names):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK_SOFT)
    axes[0].set_ylim(0, 1.25 * top)
    _pct(axes[0], decimals=1)
    axes[0].set_title("Genuine presentations rejected, by capture condition")
    overall = 100 * metrics["metrics"]["bpcer"]
    for ax in axes:
        ax.axhline(overall, color=INK_MUTED, linewidth=1.0)
    axes[-1].text(1.0, overall, f" overall {overall:.1f}%", transform=axes[-1].get_yaxis_transform(), fontsize=8.5, color=INK_SOFT, va="center")
    return save(fig, out)


def figure_depth(signals: dict, out: Path) -> Path:
    apply_style()
    depth = signals["depth"]
    real, flat = np.array(depth["samples"]["real"]), np.array(depth["samples"]["flat"])
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    bins = np.linspace(0, max(0.12, float(np.percentile(real, 99))), 49)
    ax.hist(flat, bins=bins, color=ATTACK, alpha=0.9, edgecolor=SURFACE, linewidth=0.8, label=f"Photograph tilted 15 to 30 degrees, {len(flat)} pairs")
    ax.hist(np.clip(real, None, bins[-1]), bins=bins, color=CLASS_COLOURS["live"], alpha=0.9, edgecolor=SURFACE, linewidth=0.8,
            label=f"Real head seen from two cameras, {len(real)} pairs")
    ax.axvline(depth["threshold"], color=INK, linewidth=1.2)
    ax.annotate("threshold", (depth["threshold"], ax.get_ylim()[1]), xytext=(5, -4), textcoords="offset points", va="top", fontsize=9, color=INK_SOFT)
    ax.set_xlabel("landmark motion a flat surface cannot explain, in eye distances")
    ax.set_ylabel("pairs")
    ax.set_title("Depth from motion separates real heads from flat media")
    ax.legend(loc="upper right")
    return save(fig, out)


def figure_pulse(cfg: Config, out: Path) -> Path | None:
    """Pulse waveform and spectrum from real footage with an injected heartbeat of known rate."""
    from ..signals.rppg import PulseEstimator
    from ..vision.tracker import FaceTracker
    from .signals import inject_pulse

    video = cfg.path("paths.video_dir") / "head_pose_face_detection_female.mp4"
    if not video.exists():
        return None
    apply_style()
    tracker = FaceTracker(cfg)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(26 * fps))
    frames = [cap.read()[1] for _ in range(int(12 * fps))]
    cap.release()
    observations = [tracker.process(f, i / fps) for i, f in enumerate(frames)]
    rng = np.random.default_rng(0)
    results = {}
    for name, amplitude in (("Footage as recorded", 0.0), ("With an injected 84 per minute heartbeat of 0.6% amplitude", 0.006)):
        estimator = PulseEstimator.from_config(cfg)
        for i, (frame, obs) in enumerate(zip(frames, observations)):
            if obs is None:
                continue
            t = i / fps
            wave = np.sin(2 * np.pi * 1.4 * t) + 0.3 * np.sin(4 * np.pi * 1.4 * t + 0.8)
            estimator.add(t, inject_pulse(frame, obs.landmarks, float(wave), amplitude, rng) if amplitude else frame, obs.landmarks)
        results[name] = estimator.estimate()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 5.6), gridspec_kw={"hspace": 0.55, "wspace": 0.2})
    for row, (name, result) in enumerate(results.items()):
        colour = SERIES[0] if row else INK_MUTED
        t = np.arange(len(result.waveform)) / result.sample_rate
        axes[row, 0].plot(t, result.waveform / (np.abs(result.waveform).max() + 1e-9), color=colour, linewidth=1.6)
        axes[row, 0].set_title(name, fontsize=10.5)
        axes[row, 0].set_xlabel("seconds")
        axes[row, 0].set_yticks([])
        band = result.freqs <= 3.2
        axes[row, 1].plot(60 * result.freqs[band], result.power[band] / result.power[band].max(), color=colour, linewidth=1.6)
        axes[row, 1].set_xlabel("beats per minute")
        axes[row, 1].set_yticks([])
        axes[row, 1].set_title(f"spectrum: peak at {result.bpm:.0f} per minute, signal to noise {result.snr_db:+.1f} dB", fontsize=10.5)
    fig.suptitle("Remote pulse: skin colour signal and its spectrum", x=0.125, ha="left", fontsize=12.5, fontweight="bold", y=1.0)
    return save(fig, out)


def figure_blinks(reports: Path, out: Path) -> Path | None:
    trace = _load(reports / "blink_trace.json")
    if not trace:
        return None
    apply_style()
    t, ratio = np.array(trace["times"]), np.array(trace["ratio"])
    window = (t >= 60) & (t <= 90)
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(t[window], ratio[window], color=SERIES[0], linewidth=1.5)
    for blink in trace["blinks"]:
        if 60 <= blink["start"] <= 90:
            ax.axvspan(blink["start"], blink["end"], color=SERIES[1], alpha=0.28, linewidth=0)
    ax.axhline(0.72, color=INK_MUTED, linewidth=1.0)
    ax.text(90.2, 0.72, "closed below", fontsize=8.5, color=INK_SOFT, va="center")
    ax.set_xlim(60, 90)
    ax.set_ylim(0.3, 1.5)
    ax.set_xlabel("seconds into the clip")
    ax.set_ylabel("eye opening relative\nto the user's own baseline")
    ax.set_title("Blink detection on real footage, shaded spans are detected blinks")
    return save(fig, out)


def figure_sessions(reports: Path, out: Path) -> Path | None:
    summary = _load(reports / "sessions.json")
    table_path = reports / "session_results.csv"
    if summary is None or not table_path.exists():
        return None
    apply_style()
    table = pd.read_csv(table_path)
    table["frame_scores"] = table["frame_scores"].map(json.loads)
    names = ["live"] + SPECIES_ORDER
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        rows = table[table["species"] == name]
        if rows.empty:
            continue
        session_means = rows["frame_scores"].map(lambda v: float(np.mean(v)) if len(v) else np.nan).dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(session_means)), session_means, s=46, color=CLASS_COLOURS[name], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.axhline(summary["threshold"], color=INK, linewidth=1.2)
    ax.text(len(names) - 0.55, summary["threshold"] + 0.012, "balanced threshold", fontsize=9, color=INK_SOFT, va="bottom", ha="right")
    ax.set_xticks(range(len(names)), [CLASS_LABELS[n].replace(", ", ",\n") for n in names])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("mean passive score of the session")
    ax.set_title("Real video of a person the model never saw: every dot is one ten second session")
    ax.grid(axis="x", visible=False)
    return save(fig, out)


STAGE_LABELS = {
    "ablation_1_baseline": "1\nfirst\nmodel",
    "ablation_2_invariant_inputs": "2\ninvariant\ninputs",
    "ablation_3_second_domain": "3\nsecond\ncamera",
    "ablation_4_clutter": "4\nbackground\nclutter",
    "shipped": "5\nthird camera\n(shipped)",
}


def figure_domain_gap(reports: Path, out: Path) -> Path | None:
    """Genuine people from cameras outside the training data, for every stage of the model."""
    data = _load(reports / "domain_gap.json")
    if data is None:
        return None
    apply_style()
    stages = [name for name in STAGE_LABELS if name in data["models"]]
    clips = list(data["clips"])
    fig, axes = plt.subplots(1, len(clips), figsize=(6.6 * len(clips), 4.3), sharey=True, gridspec_kw={"wspace": 0.07})
    for ax, clip in zip(np.atleast_1d(axes), clips):
        values, colours, notes = [], [], []
        for name in stages:
            entry = data["models"][name]["clips"][clip]
            values.append(100 * entry["rejected_at_eer_threshold"])
            seen = entry.get("clip_used_in_training", False)
            colours.append(INK_MUTED if seen else SERIES[0])
            notes.append("in training" if seen else "")
        x = np.arange(len(stages))
        ax.bar(x, values, width=0.5, color=colours)
        for xi, v, n in zip(x, values, notes):
            ax.text(xi, v + 1.5, f"{v:.0f}%" + (f"\n{n}" if n else ""), ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(x, [STAGE_LABELS[s] for s in stages], fontsize=9.5)
        ax.set_title(clip[0].upper() + clip[1:] + f", {data['clips'][clip]['frames']} frames", fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 118)
    np.atleast_1d(axes)[0].set_ylabel("genuine frames rejected")
    _pct(np.atleast_1d(axes)[0])
    fig.suptitle("Real people rejected by the passive model at each stage of its development", x=0.125, ha="left", fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.125, -0.07, "Each stage keeps the changes of the stages before it. Every model is judged at its own balanced threshold from validation data.",
             fontsize=8.5, color=INK_MUTED)
    return save(fig, out)


def make_all_figures(cfg: Config, log=print) -> list[Path]:
    """Rebuild every figure for which the inputs exist."""
    reports = cfg.path("paths.reports_dir")
    figures = cfg.path("paths.figures_dir")
    images = PROJECT_ROOT / "docs" / "images"
    made: list[Path] = []

    def done(path):
        if path is not None:
            made.append(path)
            log(f"  wrote {path}")

    done(figure_architecture(images / "architecture.png"))
    model = None
    if cfg.path("assets.passive_model").exists():
        from ..models.inference import PassiveLiveness

        model = PassiveLiveness(cfg)
    done(figure_attack_gallery(cfg, images / "attack_gallery.png", model))
    done(figure_hud_sequence(cfg, images / "session_screens.png"))
    done(figure_gradcam(cfg, images / "explanations.png"))
    meta_path = cfg.path("assets.passive_meta")
    if meta_path.exists():
        done(figure_training(json.loads(meta_path.read_text()), figures / "training_curves.png"))
    metrics = _load(reports / "metrics.json")
    if metrics is not None and (reports / "test_scores.csv").exists():
        scores = pd.read_csv(reports / "test_scores.csv")
        baseline = pd.read_csv(reports / "baseline_scores.csv") if (reports / "baseline_scores.csv").exists() else None
        done(figure_score_distribution(scores, metrics["threshold"], figures / "score_distribution.png"))
        done(figure_det(scores, baseline, metrics, figures / "det_curves.png"))
        done(figure_species(metrics, _load(reports / "unseen.json"), figures / "apcer_by_species.png"))
        done(figure_confusion(metrics, figures / "species_confusion.png"))
        done(figure_robustness(metrics, figures / "robustness.png"))
        done(figure_slices(metrics, figures / "bpcer_by_condition.png"))
    signals = _load(reports / "signals.json")
    if signals is not None and "depth" in signals:
        done(figure_depth(signals, figures / "depth_cue.png"))
    done(figure_pulse(cfg, figures / "pulse.png"))
    done(figure_blinks(reports, figures / "blink_trace.png"))
    done(figure_sessions(reports, figures / "video_sessions.png"))
    done(figure_domain_gap(reports, figures / "domain_gap.png"))
    return made
