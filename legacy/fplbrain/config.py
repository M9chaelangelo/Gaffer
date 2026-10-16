"""Configuration loading.

Two YAML files drive the engine: ``config/rules.yaml`` describes the game itself
(scoring, chips, squad constraints) and ``config/settings.yaml`` describes the
user (entry id, horizons, data sources). Either can be shadowed by a
``*.local.yaml`` sibling that is git-ignored, so personal settings and secrets
never end up in a commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

POSITIONS = ("GKP", "DEF", "MID", "FWD")

# FPL's element_type ids are stable across seasons.
ELEMENT_TYPE_TO_POSITION = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_TO_ELEMENT_TYPE = {v: k for k, v in ELEMENT_TYPE_TO_POSITION.items()}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_yaml_with_local(name: str, config_dir: Path | None = None) -> dict:
    """Load ``<name>.yaml`` merged with an optional ``<name>.local.yaml``."""
    directory = config_dir or CONFIG_DIR
    base = _load_yaml(directory / f"{name}.yaml")
    local = _load_yaml(directory / f"{name}.local.yaml")
    return _deep_merge(base, local)


SECRETS_FILENAME = "secrets.env"


def load_secrets(config_dir: Path | None = None) -> list[str]:
    """Load ``config/secrets.env`` into the environment; return the keys found.

    Secrets live in one git-ignored file of ``KEY=value`` lines rather than in
    the YAML, so that nothing sensitive can be committed by editing a config
    file. A value already present in the real environment always wins, which
    keeps CI and one-off overrides working.
    """
    path = (config_dir or CONFIG_DIR) / SECRETS_FILENAME
    if not path.exists():
        return []

    loaded: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        loaded.append(key)
        os.environ.setdefault(key, value)
    return loaded


@dataclass
class Rules:
    """The rules of the game, as loaded from ``rules.yaml``."""

    raw: dict = field(default_factory=dict)

    # -- squad ---------------------------------------------------------------
    @property
    def budget(self) -> float:
        return float(self.raw["squad"]["budget"])

    @property
    def squad_size(self) -> int:
        return int(self.raw["squad"]["size"])

    @property
    def squad_positions(self) -> dict[str, int]:
        return dict(self.raw["squad"]["positions"])

    @property
    def max_per_club(self) -> int:
        return int(self.raw["squad"]["max_per_club"])

    @property
    def formation_bounds(self) -> dict[str, tuple[int, int]]:
        return {k: tuple(v) for k, v in self.raw["squad"]["formation"].items()}

    # -- transfers -----------------------------------------------------------
    @property
    def hit_cost(self) -> int:
        return int(self.raw["transfers"]["hit_cost"])

    @property
    def max_banked_transfers(self) -> int:
        return int(self.raw["transfers"]["max_banked"])

    # -- chips ---------------------------------------------------------------
    @property
    def chip_sets(self) -> list[dict]:
        return list(self.raw["chips"]["sets"])

    def chip_window(self, chip_set_name: str) -> tuple[int, int]:
        for chip_set in self.chip_sets:
            if chip_set["name"] == chip_set_name:
                lo, hi = chip_set["gameweeks"]
                return int(lo), int(hi)
        raise KeyError(f"unknown chip set: {chip_set_name}")

    def chip_set_for_gameweek(self, gameweek: int) -> str | None:
        for chip_set in self.chip_sets:
            lo, hi = chip_set["gameweeks"]
            if lo <= gameweek <= hi:
                return str(chip_set["name"])
        return None

    @property
    def triple_captain_multiplier(self) -> int:
        return int(self.raw["chips"]["triple_captain_multiplier"])

    # -- scoring -------------------------------------------------------------
    @property
    def scoring(self) -> dict:
        return self.raw["scoring"]

    @property
    def bps_weights(self) -> dict:
        return self.raw["bps"]["weights"]

    def defcon_rule(self, position: str) -> dict | None:
        """Threshold rule for defensive contribution points, or None."""
        return self.raw["scoring"]["defensive_contribution"]["thresholds"].get(position)

    @property
    def defcon_points(self) -> int:
        return int(self.raw["scoring"]["defensive_contribution"]["points"])


@dataclass
class Settings:
    """User settings, as loaded from ``settings.yaml``."""

    raw: dict = field(default_factory=dict)
    repo_root: Path = REPO_ROOT

    @property
    def entry_id(self) -> int | None:
        value = self.raw.get("manager", {}).get("entry_id")
        env = os.environ.get("FPL_ENTRY_ID")
        if env:
            return int(env)
        return int(value) if value else None

    @property
    def cookie(self) -> str | None:
        """The FPL session cookie, from the environment or a local file."""
        env = os.environ.get("FPL_COOKIE")
        if env:
            return env.strip()
        rel = self.raw.get("manager", {}).get("cookie_file")
        if not rel:
            return None
        path = self.repo_root / rel
        if path.exists():
            text = path.read_text().strip()
            return text or None
        return None

    @property
    def planning_horizon(self) -> int:
        return int(self.raw.get("horizon", {}).get("planning", 6))

    @property
    def strategy_horizon(self) -> int:
        return int(self.raw.get("horizon", {}).get("strategy", 12))

    @property
    def cache_dir(self) -> Path:
        return self.repo_root / self.raw.get("data", {}).get("cache_dir", "data/cache")

    @property
    def db_path(self) -> Path:
        return self.repo_root / self.raw.get("data", {}).get("db_path", "data/fplbrain.sqlite")

    def cache_ttl(self, kind: str, default: int = 3600) -> int:
        return int(self.raw.get("data", {}).get("cache_ttl", {}).get(kind, default))

    @property
    def odds(self) -> dict:
        return self.raw.get("odds", {})

    @property
    def odds_api_key(self) -> str | None:
        return os.environ.get("ODDS_API_KEY") or None

    @property
    def model(self) -> dict:
        return self.raw.get("model", {})

    @property
    def simulation(self) -> dict:
        return self.raw.get("simulation", {})

    @property
    def risk(self) -> dict:
        return self.raw.get("risk", {})

    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a dotted path out of the raw settings, e.g. ``"model.blend"``."""
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


@lru_cache(maxsize=1)
def get_rules() -> Rules:
    return Rules(load_yaml_with_local("rules"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_secrets()
    return Settings(load_yaml_with_local("settings"))


def reset_config_cache() -> None:
    """Drop memoised config — used by tests and after editing settings."""
    get_rules.cache_clear()
    get_settings.cache_clear()
