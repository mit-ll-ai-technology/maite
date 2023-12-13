# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import maite

# -- Project information -----------------------------------------------------

project = "maite"
copyright = "2023 Massachusetts Institute of Technology"
author = "Ryan Soklaski, Justin Goodwin, Michael Yee"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""


# -- General configuration ---------------------------------------------------

REPO_URL = "https://gitlab.jatic.net/jatic/cdao/maite"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_codeautolink",
    "myst_parser",
]

autosummary_generate = False
numpydoc_show_inherited_class_members = False

# Strip input prompts:
# https://sphinx-copybutton.readthedocs.io/en/latest/#strip-and-configure-input-prompts-for-code-cells
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


default_role = "py:obj"

autodoc_typehints = "none"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html
# A dictionary of external sites
#   alias ->  (base-URL, prefix)
extlinks = {
    "commit": (REPO_URL + "commit/%s", "commit %s"),
    "gh-file": (REPO_URL + "blob/master/%s", "%s"),
    "gh-link": (REPO_URL + "%s", "%s"),
    "issue": (REPO_URL + "issues/%s", "issue #%s"),
    "pull": (REPO_URL + "pull/%s", "pull request #%s"),
    "plymi": ("https://www.pythonlikeyoumeanit.com/%s", "%s"),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
# html_logo = "../../brand/maite_logo_full_light_blue.png"

html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
    # "favicons": [
    #     {
    #         "rel": "icon",
    #         "sizes": "32x32",
    #         "href": "maite_favicon_32x32.png",
    #     },
    #     {
    #         "rel": "icon",
    #         "sizes": "64x64",
    #         "href": "maite_favicon_64x64.png",
    #     },
    # ],
    # "icon_links": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/mit-ll-responsible-ai/maite",
    #         "icon": "fab fa-github-square",
    #     },
    # ],
}


def setup(app):
    app.add_js_file(
        "https://www.googletagmanager.com/gtag/js?id=UA-115029372-2",
        loading_method="async",
    )
    app.add_js_file("gtag.js")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

myst_heading_anchors = 3
