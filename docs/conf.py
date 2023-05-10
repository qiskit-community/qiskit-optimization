# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

"""
Sphinx documentation builder
"""

import qiskit_sphinx_theme
import qiskit_optimization
from custom_directives import (
    IncludeDirective,
    GalleryItemDirective,
    CustomGalleryItemDirective,
    CustomCalloutItemDirective,
    CustomCardItemDirective,
)

# Set env flag so that we can doc functions that may otherwise not be loaded
# see for example interactive visualizations in qiskit.visualization.
os.environ["QISKIT_DOCS"] = "TRUE"

# -- Project information -----------------------------------------------------
project = "Qiskit Optimization"
copyright = f"2018, {date.today().year}, Qiskit Optimization Development Team"  # pylint: disable=redefined-builtin
author = "Qiskit Optimization Development Team"

# The short X.Y version
version = qiskit_optimization.__version__
# The full version, including alpha/beta/rc tags
release = qiskit_optimization.__version__

rst_prolog = """
.. raw:: html

    <br><br><br>

.. |version| replace:: {0}
""".format(
    release
)

nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::
        This page was generated from `docs/{{ docname }}`__.

        __"""

vers = version.split(".")
link_str = f" https://github.com/Qiskit/qiskit-optimization/blob/stable/{vers[0]}.{vers[1]}/docs/"
nbsphinx_prolog += link_str + "{{ docname }}"

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_design",
    "jupyter_sphinx",
    "reno.sphinxext",
    "sphinx.ext.doctest",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "qiskit_sphinx_theme",
]
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["style.css", "custom.css", "gallery.css"]

nbsphinx_timeout = 180
nbsphinx_execute = os.getenv("QISKIT_DOCS_BUILD_TUTORIALS", "never")
nbsphinx_widgets_path = ""
nbsphinx_thumbnails = {
    "tutorials/01_quadratic_program": "_static/1_quadratic_program.png",
    "tutorials/02_converters_for_quadratic_programs": "_static/2_converters.png",
    "tutorials/03_minimum_eigen_optimizer": "_static/3_min_eig_opt.png",
    "tutorials/04_grover_optimizer": "_static/4_grover.png",
    "tutorials/05_admm_optimizer": "_static/5_ADMM.png",
    "**": "_static/no_image.png",
}

spelling_word_list_filename = "../.pylintdict"
spelling_filters = ["lowercase_filter.LowercaseFilter"]

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = False

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------
# Move type hints from signatures to the parameter descriptions (except in overload cases, where
# that's not possible).
autodoc_typehints = "description"
# Only add type hints from signature to description body if the parameter has documentation.  The
# return type is always added to the description (if in the signature).
autodoc_typehints_description_target = "documented_params"

autodoc_default_options = {
    "inherited-members": None,
}

autoclass_content = "both"

# If true, figures, tables and code-blocks are automatically numbered if they
# have a caption.
numfig = True

# A dictionary mapping 'figure', 'table', 'code-block' and 'section' to
# strings that are used for format of figure numbers. As a special character,
# %s will be replaced to figure number.
numfig_format = {"table": "Table %s"}
# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# For Adding Locale
locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False  # optional.

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**site-packages", "_build", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "colorful"

# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined), e.g. for
# py:function directives.
add_module_names = False

# A list of prefixes that are ignored for sorting the Python module index
# (e.g., if this is set to ['foo.'], then foo.bar is shown under B, not F).
# This can be handy if you document a project that consists of a single
# package. Works only for the HTML builder currently.
modindex_common_prefix = ["qiskit_optimization."]

# -- Configuration for extlinks extension ------------------------------------
# Refer to https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#
html_theme = "qiskit_sphinx_theme"

html_theme_path = [".", qiskit_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "docplex.mp": ("https://ibmdecisionoptimization.github.io/docplex-doc/mp", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
}

html_context = {"analytics_enabled": True}
# -- Extension configuration -------------------------------------------------


def setup(app):
    app.add_directive("includenodoc", IncludeDirective)
    app.add_directive("galleryitem", GalleryItemDirective)
    app.add_directive("customgalleryitem", CustomGalleryItemDirective)
    app.add_directive("customcarditem", CustomCardItemDirective)
    app.add_directive("customcalloutitem", CustomCalloutItemDirective)
    app.setup_extension("versionutils")
