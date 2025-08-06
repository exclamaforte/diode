Documentation Guide
=================

This guide explains how to write docstrings for Sphinx, how to build the documentation, and how to deploy it.

Writing Docstrings
-----------------

The project is configured to use Google-style docstrings, which are parsed by Sphinx using the napoleon extension. Here's a quick overview of how to write Google-style docstrings:

Class Docstrings
~~~~~~~~~~~~~~~

.. code-block:: python

    class MyClass:
        """Summary of the class.

        More detailed description of the class.

        Attributes:
            attr1 (type): Description of attr1.
            attr2 (type): Description of attr2.
        """

Method Docstrings
~~~~~~~~~~~~~~~

.. code-block:: python

    def my_method(self, param1, param2):
        """Summary of the method.

        More detailed description of the method.

        Args:
            param1 (type): Description of param1.
            param2 (type): Description of param2.

        Returns:
            type: Description of the return value.

        Raises:
            ExceptionType: When and why this exception is raised.
        """

Function Docstrings
~~~~~~~~~~~~~~~~

.. code-block:: python

    def my_function(param1, param2):
        """Summary of the function.

        More detailed description of the function.

        Args:
            param1 (type): Description of param1.
            param2 (type): Description of param2.

        Returns:
            type: Description of the return value.

        Raises:
            ExceptionType: When and why this exception is raised.
        """

Examples
~~~~~~~

For more detailed examples, see the :doc:`docstring_examples` module.

.. literalinclude:: docstring_examples.py
   :language: python
   :linenos:

Building Documentation
--------------------

To build the documentation, you can use the provided script:

.. code-block:: bash

    ./build_docs.sh

This will:

1. Check if Sphinx is installed in your current Python environment
2. Install the documentation dependencies if needed (with your confirmation)
3. Build the HTML documentation
4. Open the documentation in your browser if possible

Alternatively, you can build the documentation manually:

.. code-block:: bash

    # Make sure you're in your preferred Python environment (venv, conda, etc.)

    # Install documentation dependencies if needed
    pip install -e ".[docs]"

    # Build the documentation
    cd docs
    make html

    # View the documentation
    open _build/html/index.html  # On macOS
    # or
    xdg-open _build/html/index.html  # On Linux

Live Preview
~~~~~~~~~~

You can use the `sphinx-autobuild` extension to automatically rebuild the documentation when you make changes:

.. code-block:: bash

    cd docs
    make livehtml

This will start a local server at http://localhost:8000 that will automatically update when you make changes to the documentation.

Deploying Documentation
--------------------

There are several options for deploying the documentation:

GitHub Pages
~~~~~~~~~~

If your project is hosted on GitHub, you can use GitHub Pages to host your documentation:

1. Build the documentation:

   .. code-block:: bash

       ./build_docs.sh

2. Create a `gh-pages` branch:

   .. code-block:: bash

       git checkout --orphan gh-pages
       git rm -rf .
       touch .nojekyll
       git add .nojekyll
       git commit -m "Initial gh-pages commit"
       git push origin gh-pages

3. Copy the built documentation to the `gh-pages` branch:

   .. code-block:: bash

       git checkout main
       cd docs
       make html
       cp -r _build/html/* /tmp/
       git checkout gh-pages
       cp -r /tmp/* .
       git add .
       git commit -m "Update documentation"
       git push origin gh-pages

Read the Docs
~~~~~~~~~~

You can also use Read the Docs to host your documentation:

1. Create an account on https://readthedocs.org/
2. Connect your GitHub repository
3. Configure the project to use the `docs` directory
4. Read the Docs will automatically build and host your documentation

Custom Web Server
~~~~~~~~~~~~~

You can also deploy the documentation to any web server:

1. Build the documentation:

   .. code-block:: bash

       ./build_docs.sh

2. Copy the contents of the `docs/_build/html` directory to your web server.

Adding New Documentation
---------------------

To add new documentation:

1. Create a new `.rst` or `.md` file in the `docs` directory
2. Add the file to the table of contents in `index.rst`
3. Build the documentation to see the changes

For example, to add a new file called `advanced_usage.rst`:

.. code-block:: rst

    Advanced Usage
    =============

    This is the advanced usage guide.

    ...

Then add it to `index.rst`:

.. code-block:: rst

    .. toctree::
       :maxdepth: 2
       :caption: Contents:

       readme
       data
       data_gathering
       api/index
       todo
       docstring_guide
       advanced_usage
