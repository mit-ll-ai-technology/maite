# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT

import shutil
import sys
from pathlib import Path
from typing import Sequence


def latest_version(versions: Sequence[str]):
    # Remove leading v from v0.6.0, v0.5.0, etc
    vs = []
    for v in versions:
        if v[0] != "v":
            print(f"Could not parse version: {v}", file=sys.stderr)
            continue

        parts = v[1:].split(".")
        if len(parts) != 3:
            print(f"Could not parse version: {v}", file=sys.stderr)
            continue
        try:
            vs.append((v, tuple(map(int, parts))))
        except Exception:
            print(f"Could not parse version: {v}", file=sys.stderr)
            continue
    if len(vs) == 0:
        raise ValueError(f"No versions found given, {versions}")

    max_version = version, t = max(vs, key=lambda versionstr_parts: versionstr_parts[1])
    return max_version[0]


def stable_html_versions():
    if not Path("stable_html").exists():
        return []
    return [d for d in Path("stable_html").iterdir() if d.is_dir()]


def copy_files_in_src_to_dst(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)


def copy_to_documentation_to_dir(dest_dir: Path):
    assert Path().cwd().name == "docs", "Must be run from docs directory"
    shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True)

    src_latest = Path("../.tox/docs/build/html")

    if src_latest.exists():
        dest_latest = dest_dir / "latest"
        print(f"Copying latest documentation to {dest_latest}", file=sys.stderr)
        shutil.copytree(src=src_latest, dst=dest_latest)

    if len(stable_html_versions()) == 0:
        return

    for old_version_path in stable_html_versions():
        version = old_version_path.name
        dest_version = dest_dir / version
        print(f"Copying {version} documentation to {dest_version}", file=sys.stderr)
        shutil.copytree(src=old_version_path, dst=dest_version)

    latest_stable_version = latest_version([p.name for p in stable_html_versions()])
    src_latest_stable_version = Path("stable_html") / latest_stable_version
    print(
        f"Copying last stable version ({latest_stable_version}) documentation to {dest_dir}",
        file=sys.stderr,
    )
    copy_files_in_src_to_dst(src_latest_stable_version, dest_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(len(sys.argv))
        print("Usage: python3 copy_docs_to_dir <dir>", file=sys.stderr)
        sys.exit(1)
    destdir = Path(sys.argv[1])
    copy_to_documentation_to_dir(destdir)
