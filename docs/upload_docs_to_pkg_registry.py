# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT

"""
upload_docs_to_pkg_registry.py

This file handles uploading MAITE release documentation to the gitlab generic package
registry. It is intended to be called in the gitlab CI pages job when the commit has
a release version tag (vN.N.N).

This file used for three different purposes 1. Creating a MAITE documentation file for
later manual upload. 2. Uploading a previously create MAITE documentation file. 3. Building
and uploading a MAITE documentation file.

When manually uploading a file a version must be given. In CI the --skip-nonrelease option
is used abort uploading document if a git release tag is not present.


Usage:
    python docs/upload_docs_to_pkg_registry.py upload --skip-nonrelease --docdir .tox/docs/build/html --token $CI_DOC_TOKEN
    python docs/upload_docs_to_pkg_registry.py upload_file --version v1.2.3 --file maite-doc-v1.2.3.tar.gz --token $CI_DOC_TOKEN
    python docs/upload_docs_to_pkg_registry.py build --version v1.2.3 --outdir $OUT_DIR --docdir .tox/docs/build/html

Arguments:
    --skip-nonrelease   Do nothing if a git release version tag is not present.
    --docdir            Directory containing generated documentation
    --version           Release version in vN.N.N format
    --outdir            Directory where maite-doc-$VERSION.tar.gz will be generated.
    --token             Gitlab generic package registry key
"""

from __future__ import annotations

import io
import logging
import re
import subprocess
import sys
import tarfile
from http import HTTPStatus
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


BASE_URL = "https://gitlab.jatic.net/api/v4/projects"
PROJECT_ID = "70"


def get_package_url(version: str):
    PACKAGE_NAME = "maite-doc"
    filename = artifact_filename(version)
    return (
        f"{BASE_URL}/{PROJECT_ID}/packages/generic/{PACKAGE_NAME}/{version}/{filename}"
    )


def artifact_filename(version):
    validate_version(version)
    return f"maite-doc-{version}.tar.gz"


def upload_package(file: Path | bytes, package_url: str, token: str):
    headers = {"Private-Token": token}

    if isinstance(file, str) or isinstance(file, Path):
        file_path = Path(file)
        logging.info(f"Reading {file_path}")
        file_data = file_path.read_bytes()
    else:
        file_data = file

    response = requests.put(package_url, headers=headers, data=file_data)

    if response.status_code == HTTPStatus.CREATED:
        logger.info(f"Uploaded to {package_url}")
    else:
        logger.error(
            f"Could not upload to {package_url}. Status code: {response.status_code}\nResponse: {response.text}"
        )


def validate_version(version: str):
    v = version.strip()
    if v[0] != "v":
        raise ValueError(f"Invalid version: Expecting vN.N.N not: {v}")
    ns = v[1:].split(".")
    if len(ns) != 3:
        raise ValueError(f"Invalid version: Expecting vN.N.N not: {v}")
    for n in ns:
        try:
            int(n)
        except Exception:
            raise ValueError(f"Invalid version: Expecting vN.N.N not: {v}")
    return v


def validate_docdir(docdir: Path):
    if not (docdir / "index.html").exists():
        raise ValueError(f"Doc dir {docdir} docs not contain index.html")
    return docdir


def build_targz(docdir: Path, version: str) -> bytes:
    validate_docdir(docdir)
    validate_version(version)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as targz:
        for item in docdir.rglob("*"):
            targz.add(item, arcname=Path(version) / item.relative_to(docdir))
    buf.seek(0)
    return buf.read()


def get_version_tag() -> str | None:
    try:
        result = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )

        txt = result.stdout
        tags = txt.splitlines()
        release_tags: list[str] = []
        for tag in tags:
            m = re.match(r"^(v[0-9]+\.[0-9]+\.[0-9]+)$", tag.strip())
            if m:
                release_tag = m.group(1)
                release_tags.append(release_tag)
                logger.info(f"Found release tag: {release_tag}")
        if len(release_tags) > 1:
            raise ValueError(
                f"Multiple ambiguous version tags found: {' '.join(release_tags)}"
            )
        elif len(release_tags) == 1:
            return release_tags[0]
        else:
            return None
    except subprocess.CalledProcessError as e:
        logger.error("Problem finding release tag")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return None


def build_main(args):
    version: str = validate_version(args.version)
    docdir: Path = validate_docdir(args.docdir)
    outdir: Path = args.outdir
    targz_bytes = build_targz(docdir, version)
    with open(outdir / f"maite-doc-{version}.tar.gz", "wb") as fp:
        fp.write(targz_bytes)


def upload_file_main(args):
    version = validate_version(args.version)
    file: Path = args.file
    url = get_package_url(version)
    upload_package(file, url, args.token)


def build_and_upload_main(args):
    if args.skip_nonrelease:
        version = get_version_tag()
    else:
        version = validate_version(args.version)

    if version is None:
        logger.error("No release version given or found in git. Not uploading docs.")
        return

    docdir = validate_docdir(args.docdir)
    url = get_package_url(version)
    targz_bytes = build_targz(docdir, version)
    upload_package(targz_bytes, url, args.token)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    build_tar = subparsers.add_parser("build", help="Build documentation .tar.gz")
    build_tar.add_argument("--version", type=str, help="Documentation version (vX.Y.Z)")
    build_tar.add_argument(
        "--docdir", type=Path, help="Directory holding documentation (e.g. index.html)"
    )
    build_tar.add_argument("--outdir", type=Path, help="Output directory")

    upload_file = subparsers.add_parser(
        "upload_file", help="Upload a documentation .tar.gz file to gitlab artifacts"
    )
    upload_file.add_argument("--version", type=str)
    upload_file.add_argument("--file", type=Path)
    upload_file.add_argument("--token", type=str)

    upload = subparsers.add_parser(
        "upload", help="Build and upload documentation .tar.gz"
    )
    upload.add_argument("--skip-nonrelease", action="store_true")
    upload.add_argument("--version", type=str, help="Documentation version (vX.Y.Z)")
    upload.add_argument(
        "--docdir", type=Path, help="Directory holding documentation (e.g. index.html)"
    )
    upload.add_argument("--token", type=str, help="CI token")

    args = parser.parse_args()
    if args.command == "build":
        build_main(args)
    elif args.command == "upload_file":
        upload_file_main(args)
    elif args.command == "upload":
        build_and_upload_main(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
