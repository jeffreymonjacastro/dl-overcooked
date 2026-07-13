"""Build a compact layout catalog from demonstration metadata."""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import defaultdict
from typing import Any


GENERIC_DIRS = {
    "data",
    "demonstrations",
    "demostrations",
    "grabaciones",
    "recordings",
    "records",
    "record",
    "partida 1",
    "partida 2",
    "stay",
    "random_motion",
    "greedy_full_task",
    "20 grabaciones",
    "20 grabaciones completas",
}


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _layout_from_metadata(meta_path: pathlib.Path) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layout = meta.get("layout") or {}
    env = meta.get("environment") or {}
    custom = layout.get("custom_layout_dict")
    name = layout.get("layout_name") or env.get("layout_name")
    horizon = int(layout.get("horizon") or env.get("horizon") or 250)

    if custom:
        digest = hashlib.sha1(_canon(custom).encode("utf-8")).hexdigest()[:12]
        label = name or f"custom_{digest}"
        return {
            "id": f"custom:{label}:{digest}",
            "type": "custom",
            "layout_name": label,
            "custom_layout_dict": custom,
            "horizon": horizon,
            "hash": digest,
        }

    if not name:
        raise ValueError(f"Metadata lacks layout name and custom layout: {meta_path}")
    return {
        "id": f"named:{name}",
        "type": "named",
        "layout_name": name,
        "custom_layout_dict": None,
        "horizon": horizon,
        "hash": None,
    }


def _infer_layout_name(npz_path: pathlib.Path, known_names: set[str]) -> str | None:
    for part in reversed(npz_path.parts[:-1]):
        p = part.strip()
        if p in known_names:
            return p
        if p.lower() not in GENERIC_DIRS and p and p in npz_path.stem:
            return p
    return None


def build_layout_catalog(data_root: str | pathlib.Path = "data") -> dict[str, Any]:
    """Return unique layout specs weighted by distinct .npz path frequency."""
    root = pathlib.Path(data_root)
    specs: dict[str, dict[str, Any]] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    examples: defaultdict[str, list[str]] = defaultdict(list)
    missing_metadata = 0

    known_names: set[str] = set()
    for meta_path in root.rglob("*.metadata.json"):
        try:
            spec = _layout_from_metadata(meta_path)
        except Exception:
            continue
        if spec["type"] == "named":
            known_names.add(spec["layout_name"])

    for npz_path in sorted(root.rglob("*.npz")):
        meta_path = npz_path.with_name(f"{npz_path.stem}.metadata.json")
        try:
            if meta_path.exists():
                spec = _layout_from_metadata(meta_path)
            else:
                missing_metadata += 1
                inferred = _infer_layout_name(npz_path, known_names)
                if inferred is None:
                    continue
                spec = {
                    "id": f"named:{inferred}",
                    "type": "named",
                    "layout_name": inferred,
                    "custom_layout_dict": None,
                    "horizon": 250,
                    "hash": None,
                }
        except Exception:
            continue

        sid = spec["id"]
        specs.setdefault(sid, spec)
        counts[sid] += 1
        if len(examples[sid]) < 3:
            examples[sid].append(str(npz_path.relative_to(root.parent)).replace("\\", "/"))

    catalog_specs = []
    for sid, spec in specs.items():
        row = dict(spec)
        row["weight"] = int(counts[sid])
        row["example_paths"] = examples[sid]
        catalog_specs.append(row)
    catalog_specs.sort(key=lambda s: (-s["weight"], s["layout_name"], s["id"]))

    return {
        "data_root": str(root),
        "npz_count": len(list(root.rglob("*.npz"))),
        "metadata_missing_count": missing_metadata,
        "layout_count": len(catalog_specs),
        "weight_sum": int(sum(s["weight"] for s in catalog_specs)),
        "specs": catalog_specs,
    }


if __name__ == "__main__":
    print(json.dumps(build_layout_catalog(), indent=2))
