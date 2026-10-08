# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MaRN"
copyright = "2026, Arjun Manjunath"
author = "Arjun Manjunath"
release = "0.1.0"

_GITHUB_USER = "arjunmnath"
_GITHUB_REPO = "MaRN"
_GITHUB_BRANCH = "main"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.xref_missing"]

# Generate API documentation pages automatically via autosummary
autosummary_generate = True
autosummary_imported_members = False

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = f"{project} {release}"
html_short_title = project
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "github_user": _GITHUB_USER,
    "github_repo": _GITHUB_REPO,
    "github_version": _GITHUB_BRANCH,
    "doc_path": "docs",
}

html_theme_options = {
    "github_url": f"https://github.com/{_GITHUB_USER}/{_GITHUB_REPO}",
    "show_prev_next": True,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": f"https://github.com/{_GITHUB_USER}/{_GITHUB_REPO}",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/marn/",
            "icon": "fa-solid fa-box",
        },
    ],
    "logo": {
        "text": project,
    },
    "show_toc_level": 2,
    "navigation_depth": 4,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "github-dark",
}

# -- MyST Parser configuration -----------------------------------------------
# Enable anchors and common MyST extensions
myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]


# -- Options for linkcode extension -------------------------------------------
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    import sys
    import inspect
    import os

    try:
        module = sys.modules.get(info["module"])
        if module is None:
            __import__(info["module"])
            module = sys.modules[info["module"]]
    except ImportError:
        return None

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # Unwrap decorated objects if possible
    try:
        obj = inspect.unwrap(obj)
    except Exception:
        pass

    try:
        fn = inspect.getsourcefile(obj)
        if not fn:
            return None
        conf_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(conf_dir)
        rel_fn = os.path.relpath(fn, repo_root)
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    return (
        f"https://github.com/{_GITHUB_USER}/{_GITHUB_REPO}/blob/{_GITHUB_BRANCH}/{rel_fn}{linespec}"
    )
