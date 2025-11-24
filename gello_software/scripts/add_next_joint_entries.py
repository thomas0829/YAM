#!/usr/bin/env python3


"""Populate next joint fields for sequential dataset episodes.

Given a dataset directory structured as:

    /path/to/dataset/<episode_id>/<episode_id>.json

each JSON file is expected to contain a list of dict entries that already have
"left_joint" and "right_joint" fields. This script augments every dict with
"next_left_joint" and "next_right_joint" values copied from the subsequent
dict in the list. The final dict in each file uses a default resting pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, MutableMapping, Sequence


DEFAULT_JOINT_VECTOR: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
DEFAULT_JOINT_STRING: str = json.dumps(DEFAULT_JOINT_VECTOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add next joint entries to every JSON episode in a dataset directory."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Absolute path to the dataset directory (e.g. /path/to/fold_towel_dp_dataset)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files and report changes without writing them back to disk.",
    )
    return parser.parse_args()


def iter_json_files(dataset_dir: Path) -> Iterable[Path]:
    for json_path in sorted(dataset_dir.rglob("*.json")):
        if json_path.is_file():
            yield json_path


def default_like(sample_value) -> str | List[float]:
    if isinstance(sample_value, list):
        return DEFAULT_JOINT_VECTOR.copy()
    if isinstance(sample_value, str):
        return DEFAULT_JOINT_STRING
    return DEFAULT_JOINT_VECTOR.copy()


def clone_joint_value(value):
    if isinstance(value, list):
        return list(value)
    return value


def add_next_joint_fields(
    entries: Sequence[MutableMapping[str, object]],
) -> bool:
    if not entries:
        return False

    sample_left = entries[0].get("left_joint")
    sample_right = entries[0].get("right_joint")
    default_left = default_like(sample_left)
    default_right = default_like(sample_right)

    updated = False
    for idx, current in enumerate(entries):
        next_entry = entries[idx + 1] if idx + 1 < len(entries) else None

        if next_entry is not None:
            next_left_value = clone_joint_value(
                next_entry.get("left_joint", default_left)
            )
            next_right_value = clone_joint_value(
                next_entry.get("right_joint", default_right)
            )
        else:
            next_left_value = clone_joint_value(default_left)
            next_right_value = clone_joint_value(default_right)

        if current.get("next_left_joint") != next_left_value:
            current["next_left_joint"] = next_left_value
            updated = True

        if current.get("next_right_joint") != next_right_value:
            current["next_right_joint"] = next_right_value
            updated = True

    return updated


def process_file(json_path: Path, dry_run: bool) -> bool:
    with json_path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {json_path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected {json_path} to contain a list, found {type(data)!r}")

    changed = add_next_joint_fields(data)
    if changed and not dry_run:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4)
            handle.write("\n")

    return changed


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir.expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    changed_files = 0
    total_files = 0

    for json_file in iter_json_files(dataset_dir):
        total_files += 1
        try:
            if process_file(json_file, args.dry_run):
                changed_files += 1
                action = "Would update" if args.dry_run else "Updated"
                print(f"{action} {json_file}")
        except ValueError as exc:
            print(f"Skipping {json_file}: {exc}")

    summary_action = "would be updated" if args.dry_run else "updated"
    print(f"{changed_files}/{total_files} files {summary_action} under {dataset_dir}")


if __name__ == "__main__":
    main()


