from __future__ import annotations

import json
from pathlib import Path


def merge_config_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_config(config_path: str | Path) -> dict:
    return _load_json_config(Path(config_path), visited_paths=set())


def _load_json_config(config_path: Path, visited_paths: set[Path]) -> dict:
    resolved_path = config_path.resolve()
    if resolved_path in visited_paths:
        chain = " -> ".join(str(path) for path in [*visited_paths, resolved_path])
        raise ValueError(f"Config extends cycle detected: {chain}")

    with resolved_path.open("r", encoding="utf-8-sig") as handle:
        config_data = json.load(handle)

    extends = config_data.pop("extends", None)
    if extends is None:
        return config_data

    visited_paths.add(resolved_path)
    base_paths = [extends] if isinstance(extends, str) else list(extends)
    merged: dict = {}
    for base_path in base_paths:
        inherited = _load_json_config((resolved_path.parent / base_path).resolve(), visited_paths)
        merged = merge_config_dicts(merged, inherited)
    visited_paths.remove(resolved_path)
    return merge_config_dicts(merged, config_data)
