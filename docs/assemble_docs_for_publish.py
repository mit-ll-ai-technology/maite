# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT

"""
assemble_docs_for_publish.py


A command-line script for assembling the documentation published to gh-pages
 intended by be called in the gitlab CI pages job

The published documentation has two parts, the previous release documentation
stored in  the gitlab generic package repository and the latest documentation
built by tox. This script creates a directory structure like:

```
/           # doc root, contains the documentation for latest release
/v0.6.1     # doc for previous release
/v0.7.1     # doc for previous release
...
/latest     # documentation from latest commit, not necessarily part of a release
```

To create documentation from previous releases the gitlab generic registry is queried
and a limited number of previous releases are downloaded.


Usage:
    python docs/assemble_docs_for_publish.py --token $CI_DOC_TOKEN --outdir ./public

Arguments:
    --outdir            Directory where documentation will be copied
    --token             Gitlab generic package registry key

"""

import datetime
import logging
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import List

import requests
from typing_extensions import TypedDict

BASE_URL = "https://gitlab.jatic.net/api/v4/projects"
PROJECT_ID = "70"

MAX_MAJOR_VERSIONS = 3
MAX_MINOR_VERSIONS = 4
MAX_PATCH_VERSIONS = 1

logger = logging.getLogger(__name__)


class PkgInfo(TypedDict):
    id: int
    name: str
    version: str
    created_at: str
    tags: List[str]

    major: int  # Parsed from version, not in source json
    minor: int  # Parsed from version, not in source json
    patch: int  # Parsed from version, not in source json


class PkgFileInfo(TypedDict):
    created_at: str
    file_name: str


def version_to_tuple(v: str) -> tuple[int, int, int]:
    "Convert a version string vX.Y.Z to a tuple (X,Y,Z)"
    if v[0] != "v":
        raise ValueError("Version {v} does not start with 'v'")
    parts = v[1:].split(".")
    if len(parts) != 3:
        raise ValueError("Version {v} should be vNUMBER.NUMBER.NUMBER'")
    res = (int(parts[0]), int(parts[1]), int(parts[2]))
    return res


def copy_tox_docs_to_latest(dest_dir: Path):
    "Copy the latest built docs html to  dest/latest."
    src_latest = Path(".tox/docs/build/html")
    if src_latest.exists():
        dest_latest = dest_dir / "latest"
        logger.info(f"Copying latest documentation to {dest_latest}")
        shutil.copytree(src=src_latest, dst=dest_latest)
    else:
        logger.info("Latest tox build doc {src_latest} does not exist. Ignoring.")


def list_maite_doc_pkgs(token: str) -> List[PkgInfo]:
    """
    Get the list of maite-doc packages from the package registry.

    https://docs.gitlab.com/ee/api/packages.html#for-a-project
    """
    headers = {"Private-Token": token}
    url = f"{BASE_URL}/{PROJECT_ID}/packages?package_name=maite-doc&per_page=100"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        packages = response.json()
        docs = [p for p in packages if p["name"] == "maite-doc"]
        for d in docs:
            (major, minor, patch) = version_to_tuple(d["version"])
            d["major"] = major
            d["minor"] = minor
            d["patch"] = patch
        return docs
    else:
        raise ValueError(
            f"Could not download list of packages from {url} {response.status_code=:}"
        )


def get_pkg_file_infos(pkg: PkgInfo, token: str) -> list[PkgFileInfo]:
    """
    Get a list of the files for a package from the package registry

    https://docs.gitlab.com/ee/api/packages.html#list-package-files
    """
    headers = {"Private-Token": token}
    url = f"{BASE_URL}/{PROJECT_ID}/packages/{pkg['id']}/package_files"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        js = response.json()
        assert isinstance(js, list)
        js.sort(
            key=lambda pkg: datetime.datetime.strptime(
                pkg["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            reverse=True,
        )
        return js
    else:
        raise ValueError(
            f"Could not download of package info from {url} {response.status_code=:}"
        )


def get_pkg_file(pkg: PkgInfo, file_info: PkgFileInfo, token: str) -> str:
    """
    Download the file for the given the package and package file info from the package registry.

    https://docs.gitlab.com/ee/user/packages/generic_packages/index.html#download-package-file

    Returns
    -------
    str
        Downloaded file name
    """
    headers = {"Private-Token": token}
    package_name = pkg["name"]
    file_name = file_info["file_name"]
    package_version = pkg["version"]

    url = f"{BASE_URL}/{PROJECT_ID}/packages/generic/{package_name}/{package_version}/{file_name}"
    logger.debug(f"Downloading {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
        logger.debug(f"Downloaded {file_name} from {url}")
        return file_name
    else:
        raise ValueError(f"Could not download {url} {response.status_code=:}")


def keep_latest_docs(
    doc_infos: List[PkgInfo],
    *,
    max_major_versions: int,
    max_minor_versions: int,
    max_patch_version: int,
) -> list[PkgInfo]:
    """Ensure that we keep at most max_major_versions of the documentation,
    for each major version at most max_minor_versions of the documentation
    and for each major,minor version we keep at most max_patch_versions
    """
    keepers: list[PkgInfo] = []
    major_versions = list(set([d["major"] for d in doc_infos]))
    major_versions.sort(reverse=True)
    keep_major_versions = major_versions[:max_major_versions]

    major_version_map: dict[int, list[PkgInfo]] = defaultdict(list)
    for d in doc_infos:
        major_version = d["major"]
        if major_version in keep_major_versions:
            major_version_map[major_version].append(d)

    for _major, docs in major_version_map.items():
        minor_versions = list(set([d["minor"] for d in docs]))
        minor_versions.sort(reverse=True)
        keep_minor_versions = minor_versions[:max_minor_versions]

        minor_version_map: dict[int, list[PkgInfo]] = defaultdict(list)
        for d in docs:
            minor_version = d["minor"]
            if minor_version in keep_minor_versions:
                minor_version_map[minor_version].append(d)

        for _minor, docs in minor_version_map.items():
            docs.sort(reverse=True, key=lambda d: d["patch"])
            keepers.extend(docs[:max_patch_version])

    # Make output deterministic for testing.
    keepers.sort(reverse=True, key=lambda d: (d["major"], d["minor"], d["patch"]))

    return keepers


def test_keep_latest_docs():
    def di(txt: str) -> list[PkgInfo]:
        vs = txt.split()
        res: list[PkgInfo] = []
        for version in vs:
            (major, minor, patch) = version_to_tuple(version)
            pkg_info = PkgInfo(
                id=0,
                name="",
                version=version,
                created_at="",
                tags=[],
                major=major,
                minor=minor,
                patch=patch,
            )
            res.append(pkg_info)
        return res

    def t(doc_infos: list[PkgInfo], major: int, minor: int, patch: int):
        keepers = keep_latest_docs(
            doc_infos,
            max_major_versions=major,
            max_minor_versions=minor,
            max_patch_version=patch,
        )
        return [k["version"] for k in keepers]

    assert t(di("v1.0.1 v0.7.1 v0.7.2 v1.1.0"), 2, 1, 1) == ["v1.1.0", "v0.7.2"]
    assert t(di("v0.6.1 v0.7.1"), 1, 2, 1) == ["v0.7.1", "v0.6.1"]
    assert t(di("v0.6.1 v0.6.2"), 1, 1, 1) == ["v0.6.2"]
    assert t(di("v0.0.1"), 1, 1, 1) == ["v0.0.1"]
    assert t(di("v0.0.3 v0.0.1 v0.0.2"), 1, 1, 2) == ["v0.0.3", "v0.0.2"]
    assert t(di("v0.0.3 v0.0.1 v0.0.2"), 1, 1, 1) == ["v0.0.3"]
    assert t(di("v1.0.0 v0.0.1"), 1, 1, 1) == ["v1.0.0"]
    assert t(di("v1.0.0 v0.0.1"), 2, 1, 1) == ["v1.0.0", "v0.0.1"]
    assert t(di("v1.0.0 v1.0.1 v0.0.5 v0.0.6 v0.0.7"), 2, 1, 2) == [
        "v1.0.1",
        "v1.0.0",
        "v0.0.7",
        "v0.0.6",
    ]


def download_prev_docs_from_gitlab(output_dir: Path, token: str):
    """
    Download documentation for previous MAITE versions from gitlab
    package registry, unpack then in the output directory.
    """
    pkgs = list_maite_doc_pkgs(token)
    if len(pkgs) == 0:
        raise ValueError("Did not find any previous documentation")
    logger.debug(f"maite-doc versions found: {[p['version'] for p in pkgs]}")

    pkgs = keep_latest_docs(
        pkgs,
        max_major_versions=MAX_MAJOR_VERSIONS,
        max_minor_versions=MAX_MINOR_VERSIONS,
        max_patch_version=MAX_PATCH_VERSIONS,
    )
    logger.debug(f"maite-doc versions kept: {[p['version'] for p in pkgs]}")
    pkgs = sorted(pkgs, key=lambda pkg: version_to_tuple(pkg["version"]), reverse=True)

    if len(pkgs) == 0:
        logger.error("No previous maite-doc packages found.")
        return

    # Download and uppack previous versions of documentation
    for pkg in pkgs:
        file_infos = get_pkg_file_infos(pkg, token)
        if len(file_infos) > 0:
            file_name = get_pkg_file(pkg, file_infos[0], token=token)
            with tarfile.open(file_name, "r:gz") as file:
                logger.info(f"Copying {pkg['version']} documentation to {output_dir}")
                file.extractall(path=output_dir)
        else:
            logger.error(f"No files found for {pkg}")

    # E.g. copy DIR/v0.6.0 to DIR/ so that it appears in the documentation root
    #
    # Pkgs are sorted by version number, i=0 has greatest version
    last_release_pkg = pkgs[0]
    last_version = last_release_pkg["version"]
    release_docs = output_dir / last_version
    logger.info(f"Copying last stable {last_version} documentation to {output_dir}")
    shutil.copytree(release_docs, output_dir, dirs_exist_ok=True)


def main(token: str, output_dir: Path):
    if output_dir.exists():
        print(f"{output_dir} must not exist before running script", file=sys.stderr)
        sys.exit(1)

    assert Path().cwd().name == "maite", "Must be run from maite directory"
    output_dir.mkdir(parents=True)

    download_prev_docs_from_gitlab(output_dir=output_dir, token=token)
    copy_tox_docs_to_latest(output_dir)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.WARN, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    main(args.token, args.outdir)
