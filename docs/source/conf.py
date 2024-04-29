# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

master_doc = "index"
project = "energy_effects_t2k"
copyright = "2024, Pierre Boistier"
author = "Pierre Boistier"
release = "0.1.2"

# html_theme = "sphinx_book_theme"
html_theme = "pydata_sphinx_theme"

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
]
source_suffix = [".rst", ".md"]
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
# bibtex_reference_style = 'super'

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}

# Add any paths that contain templates here, relative to this directory.

templates_path = ["_templates"]

autosummary_generate = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_title = "Energy resolution and non-linearity effects on the neutrino oscillation probability at T2K and beyond"
html_title = "Energy effects at T2K"
html_show_sphinx = True
# nbsphinx_input_prompt = " "
# nbsphinx_prompt_width = 0
highlight_language = "none"
# nbsphinx_requirejs_path = ''

# html_style = 'friendly'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]

suppress_warnings = []

sphinx_tabs_disable_tab_closing = True
