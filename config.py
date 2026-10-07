"""Configuration loading.

The configuration is a plain YAML file turned into a small attribute dictionary.
Overrides use ``section.key=value`` strings so that the command line never needs
flags, for example ``python manage.py train train.epochs=20``.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml



def _find_root() -> Path:
    """Repository root: where ``configs/default.yaml`` and the model files live.

    The package may run from a checkout or from an installed copy. The root is
    taken from ``PULSEGATE_HOME`` if set, otherwise from the checkout the
    package sits in, otherwise from the current directory.
    """
    candidates = [Path(__file__).resolve().parents[1], Path.cwd()]
    if os.environ.get("PULSEGATE_HOME"):
        candidates.insert(0, Path(os.environ["PULSEGATE_HOME"]).expanduser())
    for candidate in candidates:
        if (candidate / "configs" / "default.yaml").exists():
            return candidate
    return candidates[0]


PROJECT_ROOT = _find_root()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG = PROJECT_ROOT / "configs" / "local.yaml"       # optional site settings, never committed


class Config(dict):
    """Dictionary with attribute access and nested conversion."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            return cls(value)
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"configuration has no key '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(copy.deepcopy(self.to_dict(), memo))

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def set_path(self, dotted: str, value: Any) -> None:
        """Set ``a.b.c`` style keys, creating intermediate sections when needed."""
        node = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = self._wrap(value)

    def path(self, dotted: str) -> Path:
        """Resolve a configured path (``section.key``) against the repository root."""
        node: Any = self
        for part in dotted.split("."):
            node = node[part]
        return resolve_path(node)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn ``["train.epochs=3", "webcam.mirror=false"]`` into a flat mapping."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key=value")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw) if raw.strip() != "" else ""
    return out


def load_config(path: str | Path | None = None, overrides: Iterable[str] | Mapping[str, Any] | None = None,
                use_local: bool = True) -> Config:
    """Load the defaults, then ``configs/local.yaml`` if present, an optional user file, then overrides."""
    with open(DEFAULT_CONFIG, "r", encoding="utf8") as handle:
        cfg = Config(yaml.safe_load(handle))
    if use_local and LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, "r", encoding="utf8") as handle:
            _merge(cfg, yaml.safe_load(handle) or {})
    if path is not None and resolve_path(path) != DEFAULT_CONFIG:
        with open(resolve_path(path), "r", encoding="utf8") as handle:
            _merge(cfg, yaml.safe_load(handle) or {})
    if overrides:
        flat = overrides if isinstance(overrides, Mapping) else parse_overrides(overrides)
        for key, value in flat.items():
            cfg.set_path(key, value)
    return cfg


def _merge(base: Config, extra: Mapping[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Config):
            _merge(base[key], value)
        else:
            base[key] = Config._wrap(value)
