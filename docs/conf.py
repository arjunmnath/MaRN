# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Mapping Networks"
copyright = "2026, Arjun Manjunath"
author = "Arjun Manjunath"
release = "0.1.0"

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

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

    return f"https://github.com/arjunmnath/marn/blob/main/{rel_fn}{linespec}"
