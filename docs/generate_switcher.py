# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT

"""Dynamic Version Switcher JSON Generator and Root Redirector for Sphinx.

This module automates the generation of a version switcher configuration file
(`switcher.json`) compatible with the `pydata-sphinx-theme` and writes a root-level
`index.html` redirection file for GitLab Pages and GitHub Pages deployments.

Architecture & Layout
---------------------
All active documentation versions reside at a uniform directory depth of exactly
1 level inside the deployment directory ("public/")::

    public/
    ├── index.html              <- Client-side HTML meta redirect to the latest tag
    ├── _static/
    │   └── switcher.json       <- Version mapping JSON file for the dropdown
    ├── latest/                 <- Development branch documentation build
    ├── v0.12.99/               <- Latest tagged semantic release (stable)
    └── v0.11.0/                <- Historical release documentation build

The switcher touches our build/publish process in two places, which reference
URLs by different rules:

1. On doc build, ``conf.py`` embeds a ``switcher.json`` location into the HTML
   (``html_theme_options["switcher"]["json_url"]``). This points into the
   ``html_static_path`` directory as a path *relative to that doc's own root*,
   so it is host-agnostic and safe to store immutably in the GitLab registry::

       json_url = "../_static/switcher.json"
       # from any page under <host>/<base>/latest/  ->  <host>/<base>/_static/switcher.json

2. After doc build, this script writes ``switcher.json`` with a URL per version,
   derived from the directory structure produced by ``assemble_docs_for_publish.py``.
   These URLs cannot be relative: the theme resolves them against ``<host>/``, so a
   relative value 404s whenever the host mounts the docs under a subdirectory (both
   GitLab and GitHub typically do). They must be root-anchored with the host base
   (supplied via ``--base-url``)::

       "url": "latest/"         # -> <host>/latest/         (404 when served under a subpath)
       "url": "/maite/latest/"  # -> <host>/maite/latest/   (correct; --base-url="/maite")
"""

import argparse
import json
import os
import re
import sys
from urllib.parse import urlparse

DEFAULT_PUBLIC_DIR = "public"

# A released-version directory is strictly 'v' + semantic version triple.
# Anchored and fully-matched so prerelease tags (e.g. 'v1.0.0rc1') are excluded.
VERSION_REGEX = re.compile(r"^v\d+\.\d+\.\d+$")


def get_version_directories(public_dir):
    """Scan the public directory for v-prefixed version subdirectories.

    Searches ``public_dir`` for directories whose names are exactly a
    ``v``-prefixed semantic version triple (e.g. 'v1.0.0', 'v0.12.99'),
    returning them sorted by semantic version in descending order (highest
    first). Prerelease directories (e.g. 'v1.0.0rc1') are intentionally not
    matched.

    Parameters
    ----------
    public_dir : str
        Deployment directory to scan.

    Returns
    -------
    list of str
        Matching directory names, highest version first.
    """
    versions = [e.name for e in os.scandir(public_dir) if e.is_dir() and VERSION_REGEX.match(e.name)]

    # Natural sort key: extract the three integer components for correct ordering.
    def natural_sort_key(s):
        return list(map(int, re.findall(r"\d+", s)))

    versions.sort(key=natural_sort_key, reverse=True)
    return versions


def build_switcher_data(public_dir, base_url):
    """Build the switcher entry list for the versions present in ``public_dir``.

    Parameters
    ----------
    public_dir : str
        Deployment directory to scan for version builds.
    base_url : str
        Host base path to prefix every entry URL with. May be a full URL or a
        bare path (e.g. ``https://host/maite``, ``/maite``, or ``""`` for a
        root-hosted site); it is reduced to a root-anchored path below.

    Returns
    -------
    list of dict
        Switcher entries. A ``latest (dev)`` entry is included only when a
        ``latest/`` directory actually exists. The highest release is labeled
        ``(stable)``.
    """
    # Reduce base_url to a root-anchored path with no trailing slash, keeping just
    # the path component so CI can pass GitLab's $CI_PAGES_URL verbatim. Accepts
    # either a full URL (e.g. "https://host/org-name/maite" -> "/org-name/maite")
    # or a bare path (e.g. "/maite" -> "/maite"); "" stays "" for a host-root site.
    base = urlparse(base_url).path.rstrip("/")
    switcher_data = []

    # 1. Latest/dev branch entry -- only if the build is actually present.
    if os.path.isdir(os.path.join(public_dir, "latest")):
        switcher_data.append(
            {
                "name": "latest (dev)",
                "version": "latest",
                "url": f"{base}/latest/",
            }
        )

    # 2. Tagged release entries (the highest tag is labeled as stable).
    version_dirs = get_version_directories(public_dir)
    for i, version in enumerate(version_dirs):
        label = f"{version} (stable)" if i == 0 else version
        switcher_data.append(
            {
                "name": label,
                "version": version,
                "url": f"{base}/{version}/",
            }
        )

    return switcher_data


def build_redirect_html(target):
    """Return the client-side redirect ``index.html`` body pointing at ``target``.

    The redirect lives at the deployment root (``public/index.html``) and uses a
    path relative to itself (``./<target>/index.html``), so it is correct on every
    host regardless of base path.
    """
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "  <head>\n"
        '    <meta charset="utf-8">\n'
        f'    <meta http-equiv="refresh" content="0; url=./{target}/index.html">\n'
        f'    <link rel="canonical" href="./{target}/index.html">\n'
        "    <title>Redirecting...</title>\n"
        "  </head>\n"
        "  <body>\n"
        f'    <p>Redirecting to <a href="./{target}/index.html">{target}</a>...</p>\n'
        "  </body>\n"
        "</html>\n"
    )


def generate_switcher_json(public_dir, base_url):
    """Generate ``switcher.json`` and write the root redirect ``index.html``.

    Parameters
    ----------
    public_dir : str
        Deployment directory (must already exist and contain the assembled
        version builds).
    base_url : str
        Root-anchored host base path for entry URLs (see
        :func:`build_switcher_data`).

    Raises
    ------
    SystemExit
        If ``public_dir`` does not exist, or if it contains no documentation
        builds at all (neither ``latest/`` nor any ``v*`` release). We fail hard
        rather than silently emitting an empty/placeholder switcher.
    """
    if not os.path.isdir(public_dir):
        sys.exit(f"Error: deployment directory '{public_dir}' does not exist. Run the assemble step first.")

    switcher_data = build_switcher_data(public_dir, base_url)
    if not switcher_data:
        sys.exit(f"Error: no documentation builds found in '{public_dir}' (expected 'latest/' and/or 'v*' dirs).")

    # Write switcher.json inside public/_static/.
    output_dir = os.path.join(public_dir, "_static")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "switcher.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(switcher_data, f, indent=4)

    # Redirect the deployment root to the newest tagged release if one exists,
    # otherwise to latest/ (dev). switcher_data is non-empty here, so at least
    # one of these is present.
    version_dirs = get_version_directories(public_dir)
    redirect_target = version_dirs[0] if version_dirs else "latest"
    with open(os.path.join(public_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(build_redirect_html(redirect_target))

    # Write an empty .nojekyll at the deployment root to opt out of GitHub Pages'
    # Jekyll processing. We assume sphinx-based build to handle docs directories.
    with open(os.path.join(public_dir, ".nojekyll"), "w", encoding="utf-8") as f:
        f.write("")

    print(f"Generated switcher at '{output_path}' (base_url='{base_url}') and root redirect to '{redirect_target}'.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--outdir",
        default=DEFAULT_PUBLIC_DIR,
        help="Deployment directory containing the assembled version builds (default: 'public').",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help=(
            "Host base path prefixed onto every switcher entry URL. Accepts a full "
            "URL (e.g. GitLab's $CI_PAGES_URL) or a bare path (e.g. '/maite' for a "
            "GitHub project page); only the path component is used. Default '' serves "
            "the site at the domain root."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    generate_switcher_json(public_dir=args.outdir, base_url=args.base_url)
