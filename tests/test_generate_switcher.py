# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Tests for docs/generate_switcher.py.

The generator lives under docs/ (not an importable package), so it is loaded by
file path. These tests exercise switcher.json contents, base-url prefixing, the
"latest only if present" rule, stable labeling, and fail-hard behavior.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_GEN_PATH = Path(__file__).resolve().parents[1] / "docs" / "generate_switcher.py"
_spec = importlib.util.spec_from_file_location("generate_switcher", _GEN_PATH)
assert _spec is not None
assert _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _make_builds(root: Path, names):
    """Create version build directories (each with an index.html) under ``root``."""
    for name in names:
        d = root / name
        d.mkdir(parents=True)
        (d / "index.html").write_text("<html></html>", encoding="utf-8")


@pytest.mark.parametrize(
    ("base_url", "expected_prefix"),
    [
        ("", ""),  # root-hosted site
        ("/maite", "/maite"),  # GitHub project pages (bare path)
        ("/maite/", "/maite"),  # trailing slash stripped
        ("https://host/org-name/maite", "/org-name/maite"),  # full URL (GitLab CI_PAGES_URL)
        ("https://host/", ""),  # root-hosted fork, full URL
    ],
)
def test_entry_urls_are_root_anchored_with_base(tmp_path, base_url, expected_prefix):
    _make_builds(tmp_path, ["latest", "v0.12.99", "v0.11.0"])

    data = gen.build_switcher_data(str(tmp_path), base_url)
    urls = {entry["version"]: entry["url"] for entry in data}

    assert urls["latest"] == f"{expected_prefix}/latest/"
    assert urls["v0.12.99"] == f"{expected_prefix}/v0.12.99/"
    assert urls["v0.11.0"] == f"{expected_prefix}/v0.11.0/"


def test_highest_release_labeled_stable_and_sorted(tmp_path):
    # Deliberately unsorted, with a double-digit minor to catch lexical-sort bugs.
    _make_builds(tmp_path, ["latest", "v0.9.0", "v0.12.99", "v0.11.0"])

    data = gen.build_switcher_data(str(tmp_path), "")
    releases = [e for e in data if e["version"] != "latest"]

    # Descending semantic order, highest first and labeled stable.
    assert [e["version"] for e in releases] == ["v0.12.99", "v0.11.0", "v0.9.0"]
    assert releases[0]["name"] == "v0.12.99 (stable)"
    assert releases[1]["name"] == "v0.11.0"
    assert all("(stable)" not in e["name"] for e in releases[1:])


def test_latest_entry_only_when_latest_dir_present(tmp_path):
    _make_builds(tmp_path, ["v0.12.99"])  # no latest/

    data = gen.build_switcher_data(str(tmp_path), "")

    assert all(e["version"] != "latest" for e in data)
    assert [e["version"] for e in data] == ["v0.12.99"]


def test_prerelease_dirs_are_ignored(tmp_path):
    # Only strict vX.Y.Z triples count as releases.
    _make_builds(tmp_path, ["latest", "v1.0.0", "v1.0.0rc1", "v2", "vfoo"])

    data = gen.build_switcher_data(str(tmp_path), "")
    versions = {e["version"] for e in data}

    assert versions == {"latest", "v1.0.0"}


def test_generate_writes_switcher_and_redirect(tmp_path):
    _make_builds(tmp_path, ["latest", "v0.12.99", "v0.11.0"])

    gen.generate_switcher_json(str(tmp_path), "/maite")

    switcher = json.loads((tmp_path / "_static" / "switcher.json").read_text(encoding="utf-8"))
    assert switcher[0]["url"] == "/maite/latest/"

    # Root redirect targets the newest release, via a self-relative path.
    redirect = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "./v0.12.99/index.html" in redirect


def test_redirect_targets_latest_when_no_release(tmp_path):
    _make_builds(tmp_path, ["latest"])

    gen.generate_switcher_json(str(tmp_path), "")

    redirect = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "./latest/index.html" in redirect


def test_generate_writes_nojekyll(tmp_path):
    # .nojekyll opts the deploy out of GitHub Pages' Jekyll processing, which
    # would otherwise ignore Sphinx's _static/ (underscore-prefixed) dirs.
    _make_builds(tmp_path, ["latest", "v0.12.99"])

    gen.generate_switcher_json(str(tmp_path), "")

    nojekyll = tmp_path / ".nojekyll"
    assert nojekyll.is_file()
    assert nojekyll.read_text(encoding="utf-8") == ""


def test_fail_hard_when_outdir_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit):
        gen.generate_switcher_json(str(missing), "")


def test_fail_hard_when_no_builds(tmp_path):
    (tmp_path / "_static").mkdir()  # present but contains no version builds
    with pytest.raises(SystemExit):
        gen.generate_switcher_json(str(tmp_path), "")
