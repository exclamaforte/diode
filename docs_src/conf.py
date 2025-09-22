# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "diode"
copyright = "2023, Author Name"
author = "Author Name"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

# Try to add mermaid support if available
try:
    import sphinxcontrib.mermaid

    extensions.append("sphinxcontrib.mermaid")
except ImportError:
    # Fallback: use myst_parser's mermaid support
    pass

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# GitHub Pages and deployment compatibility
html_baseurl = ""
html_file_suffix = ".html"
html_link_suffix = ".html"

# Ensure proper path handling for deployed sites
html_copy_source = True
html_use_smartypants = True
html_use_index = True
html_split_index = False
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Configure theme options for better deployment compatibility
html_theme_options = {
    "style_external_links": False,
    "navigation_depth": 4,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#2980B9",
}

# Ensure proper static file handling
html_last_updated_fmt = "%b %d, %Y"
html_domain_indices = True
html_use_modindex = True
html_add_permalinks = "¶"
html_sidebars = {}

# -- Extension configuration -------------------------------------------------
# Napoleon settings for parsing Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST settings for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "html_image",
]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Auto-generate API documentation
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]
