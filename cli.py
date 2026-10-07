"""Command line interface.

Usage: ``python manage.py <command> [key=value ...]``

Options are written as ``key=value``. Keys with a dot override the
configuration file (``train.epochs=20``), plain keys are arguments of the
command itself (``video=clip.mp4``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import Config, load_config, parse_overrides

COMMANDS: dict[str, tuple] = {}


def command(name: str, summary: str):
    def register(func):
        COMMANDS[name] = (func, summary)
        return func
    return register


def _split_args(items: list[str]) -> tuple[dict, dict]:
    """Separate configuration overrides (dotted keys) from command arguments."""
    parsed = parse_overrides(items)
    overrides = {k: v for k, v in parsed.items() if "." in k}
    args = {k: v for k, v in parsed.items() if "." not in k}
    return overrides, args


@command("info", "show versions, model details and where things are stored")
def cmd_info(cfg: Config, args: dict) -> None:
    import cv2
    import numpy
    import onnxruntime

    from . import __version__

    print(f"PulseGate AI {__version__}")
    print(f"OpenCV {cv2.__version__}, ONNX Runtime {onnxruntime.__version__}, NumPy {numpy.__version__}")
    meta_path = cfg.path("assets.passive_meta")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        stages = len(meta.get("earlier_stages", [])) + 1
        print(f"passive model: {meta.get('run')} with {meta.get('parameters'):,} parameters, trained in {stages} stage(s)")
        print(f"operating thresholds: {json.dumps({k: round(v, 4) for k, v in meta.get('thresholds', {}).items()})}")
    else:
        print("passive model: not trained yet")
    for key in ("paths.muct_dir", "paths.dataset_dir", "paths.models_dir", "paths.reports_dir"):
        print(f"{key}: {cfg.path(key)}")


@command("download_data", "download the MUCT face database and the sample videos")
def cmd_download(cfg: Config, args: dict) -> None:
    import urllib.request

    from .data.muct import download_muct

    download_muct(cfg.path("paths.muct_dir"))
    video_dir = cfg.path("paths.video_dir")
    video_dir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/"
    names = ["head_pose_face_detection_female", "head_pose_face_detection_male", "head_pose_face_detection_female_and_male", "face_demographics_walking_and_pause"]
    for name in names:
        target = video_dir / f"{name}.mp4"
        if target.exists():
            continue
        print(f"downloading {target.name}")
        # upstream spells the names with dashes, local copies use underscores throughout
        urllib.request.urlretrieve(f"{base}{name.replace('_', '-')}.mp4", target)
    print("data is ready")


@command("build_dataset", "render the presentation attack dataset from MUCT")
def cmd_build(cfg: Config, args: dict) -> None:
    from .data.builder import build_dataset

    build_dataset(cfg, limit=int(args.get("limit", 0)))


@command("add_video", "add a clip of genuine footage as an extra training domain: video=path")
def cmd_add_video(cfg: Config, args: dict) -> None:
    from .data.video_domain import add_video_domain

    if "video" not in args:
        default = cfg.path("paths.video_dir") / "head_pose_face_detection_male.mp4"
        args = {**args, "video": str(default), "name": "intel_head_pose_male"}
    add_video_domain(cfg, str(args["video"]), name=args.get("name"), max_frames=int(args.get("max_frames", 900)))


@command("train", "train PulseGateNet, calibrate it and export ONNX")
def cmd_train(cfg: Config, args: dict) -> None:
    from .training.train import train_model

    train_model(cfg)


@command("evaluate", "test set metrics, baseline, robustness and slices")
def cmd_evaluate(cfg: Config, args: dict) -> None:
    from .evaluation.evaluate import evaluate_model

    evaluate_model(cfg, robustness_samples=int(args.get("robustness_samples", 1200)), baseline_train=int(args.get("baseline_train", 9000)))


@command("evaluate_unseen", "train without one attack family and test on it")
def cmd_unseen(cfg: Config, args: dict) -> None:
    from .evaluation.unseen import evaluate_unseen

    evaluate_unseen(cfg, epochs=int(args.get("epochs", 5)), max_train=int(args.get("max_train", 14000)))


@command("evaluate_sessions", "genuine and simulated attack sessions on real video")
def cmd_sessions(cfg: Config, args: dict) -> None:
    from .evaluation.sessions import evaluate_sessions

    evaluate_sessions(cfg, live_windows=int(args.get("live_windows", 13)), attack_windows=int(args.get("attack_windows", 5)), workers=int(args.get("workers", 2)))


@command("evaluate_domain_gap", "score genuine people filmed by an unseen camera, for every model stage")
def cmd_domain_gap(cfg: Config, args: dict) -> None:
    from .evaluation.domain_gap import evaluate_domain_gap

    evaluate_domain_gap(cfg)


@command("validate_signals", "validate depth, challenges, pulse and blinks on real data")
def cmd_signals(cfg: Config, args: dict) -> None:
    from .evaluation.signals import validate_signals

    validate_signals(cfg)


@command("benchmark", "measure the latency of every stage on this machine")
def cmd_benchmark(cfg: Config, args: dict) -> None:
    from .evaluation.benchmark import benchmark

    benchmark(cfg)


@command("reproduce", "run the whole pipeline from download to figures")
def cmd_reproduce(cfg: Config, args: dict) -> None:
    """Rebuild the shipped model and every report, in the order the project followed."""
    import copy
    import shutil

    def banner(text: str) -> None:
        print(f"\n===== {text} =====")

    banner("download_data")
    cmd_download(cfg, {})
    banner("build_dataset")
    cmd_build(cfg, {})
    video_dir = cfg.path("paths.video_dir")
    banner("add_video: first clip")
    cmd_add_video(cfg, {"video": str(video_dir / "head_pose_face_detection_male.mp4"), "name": "intel_head_pose_male"})
    banner("train: stage one")
    stage_one = copy.deepcopy(cfg)
    stage_one.train.run_name = "pulsegate_stage1"
    cmd_train(stage_one, {})
    models = cfg.path("paths.models_dir")
    experiments = models / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    for suffix in (".pt", ".onnx", ".json"):          # stage one is also stage 4 of the camera comparison
        shutil.copy(models / f"pulsegate_stage1{suffix}", experiments / f"ablation_4_clutter{suffix}")
    banner("add_video: second clip")
    cmd_add_video(cfg, {"video": str(video_dir / "face_demographics_walking_and_pause.mp4"), "name": "intel_hallway", "max_frames": 600})
    banner("train: stage two")
    stage_two = copy.deepcopy(cfg)
    stage_two.train.init_checkpoint = str(models / "pulsegate_stage1.pt")
    stage_two.train.epochs, stage_two.train.lr, stage_two.train.warmup_epochs = 4, 0.0006, 0.3
    cmd_train(stage_two, {})
    for step in ("export", "evaluate", "evaluate_unseen", "evaluate_domain_gap", "evaluate_sessions", "validate_signals", "benchmark", "figures"):
        banner(step)
        COMMANDS[step][0](cfg, {})


@command("figures", "regenerate every figure in reports/figures and docs/images")
def cmd_figures(cfg: Config, args: dict) -> None:
    from .evaluation.plots import make_all_figures

    make_all_figures(cfg)


@command("webcam", "live liveness check with your camera")
def cmd_webcam(cfg: Config, args: dict) -> None:
    from .apps.webcam import run_webcam

    run_webcam(cfg, mode=str(args.get("mode", "interactive")))


@command("analyze", "check a recorded video or a still image: video=path or image=path")
def cmd_analyze(cfg: Config, args: dict) -> None:
    import cv2

    from .engine.session import LivenessEngine

    engine = LivenessEngine(cfg)
    if "image" in args:
        image = cv2.imread(str(args["image"]))
        if image is None:
            raise SystemExit(f"cannot read image {args['image']}")
        result = engine.check_image(image)
    elif "video" in args:
        result = engine.analyze_video(str(args["video"]), max_seconds=float(args["seconds"]) if "seconds" in args else None).as_dict()
    else:
        raise SystemExit("give video=path or image=path")
    text = json.dumps(result, indent=2)
    print(text)
    if "out" in args:
        Path(str(args["out"])).write_text(text)


@command("serve", "start the HTTP service and the browser demo")
def cmd_serve(cfg: Config, args: dict) -> None:
    import uvicorn

    from .apps.api import create_app

    uvicorn.run(create_app(cfg), host=str(cfg.api.host), port=int(cfg.api.port), log_level="info")


@command("collect", "record your own captures: label=live or an attack species")
def cmd_collect(cfg: Config, args: dict) -> None:
    from .apps.webcam import run_collect

    run_collect(cfg, label=str(args.get("label", "live")), seconds=float(args.get("seconds", 20)))


@command("finetune", "adapt the model to your own captures in data/captures")
def cmd_finetune(cfg: Config, args: dict) -> None:
    from .training.finetune import finetune_model

    finetune_model(cfg, epochs=int(args.get("epochs", 4)))


@command("calibrate", "measure the model on your own captures and fit a threshold: apply=true stores it")
def cmd_calibrate(cfg: Config, args: dict) -> None:
    from .evaluation.calibrate import calibrate

    calibrate(cfg, apply=bool(args.get("apply", False)), target_apcer=float(args.get("target_apcer", 0.01)))


@command("export", "refit thresholds for the configured operating point and export ONNX again")
def cmd_export(cfg: Config, args: dict) -> None:
    from .training.train import recalibrate_model

    recalibrate_model(cfg, run=str(args["run"]) if "run" in args else None)


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["PulseGate AI", "", "usage: python manage.py <command> [key=value ...]", "", "commands:"]
    lines += [f"  {name.ljust(width)}  {summary}" for name, (_, summary) in COMMANDS.items()]
    lines += ["", "examples:", "  python manage.py webcam", "  python manage.py analyze video=clip.mp4", "  python manage.py train train.epochs=20 train.batch_size=64"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("help", "h", "?"):
        print(usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command '{name}'\n\n{usage()}")
        return 2
    try:
        overrides, args = _split_args(rest)
    except ValueError as exc:
        print(str(exc))
        return 2
    config_path = args.pop("config", None)
    cfg = load_config(config_path, overrides)
    COMMANDS[name][0](cfg, args)
    return 0
