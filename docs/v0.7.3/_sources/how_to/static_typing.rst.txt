Enable Static Type Checking
================================================

Overview
--------
In statically typed languages (such as C++ and Java), the data types of variables are known at compile time. Type 
checking during compilation will catch assignments of invalid value types to variables, preventing  
runtime errors from occurring later on during program execution.

Since Python is a dynamically typed language, type errors can potentially occur at runtime. However, Python source code 
can be statically type checked by using a combination of type annotations (using the `typing` package) and external static 
analysis tools, allowing type-related errors to be caught earlier in the development process.

In this how-to, we help you configure your environment to take advantage of static type checking support. If you are not
already familiar with type annotations, see
`Typing Python Libraries <https://typing.readthedocs.io/en/latest/guides/libraries.html>`_.

Static Typing Checking with Pyright
-----------------------------------

`Pyright <https://microsoft.github.io/pyright>`_ is a static type checker for Python that can be used from the command-line 
and also via a Visual Studio Code extension. The MAITE development team uses Pyright for static type checking during 
development and the CI/CD process.

Running Pyright as a Command-Line Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If not already installed in your virtual environment, `install pyright <https://microsoft.github.io/pyright/#/installation>`_:

``$ pip install pyright``

To manually run Pyright on some Python file:

``$ pyright some-prog.py``

For a list of all supported command-line options, see
`Pyright Command-Line Options <https://microsoft.github.io/pyright/#/command-line>`_.

Running Pyright as a VS Code Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We recommend having static type checking enabled in your IDE so that you can immediately see type-related issues 
(which will appear as red squiggles) and also mouse over variables to see what their statically inferred types are.

To enable static type checking in VS Code:

1. Install the Pylance VS Code extension.
2. Enable static type checking at the "basic" level by adding::

    {
        "python.analysis.typeCheckingMode": "basic"
    }

to your ``settings.json`` file. This can be accomplished through the Settings Editor UI:

- Access Settings via the shortcut ``Ctrl+,`` (or ``Cmd+,`` on macOS)
- Search for "type checking"
- Set "Python › Analysis: Type Checking Mode" to *basic* (which is the level that MAITE developers use)

After enabling static type checking, the following code should have a type error in VS Code::

    # Wrong type being assigned to variable x
    x:int = "abc"

Running Pyright in CI/CD Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We suggest running static type checking and evaluating type completeness with Pyright as part of your CI/CD pipeline. For the 
MAITE project, we use `tox <https://tox.wiki>`_ to orchestrate tests across different environments (and these tox jobs are run
as part of GitLab or GitHub CI/CD pipelines).

The ``[testenv:typecheck]`` section of MAITE's `tox.ini <https://github.com/mit-ll-ai-technology/maite/blob/main/tox.ini>`_ file
contains directives for performing static type checking on MAITE's source code and measuring type completeness.

MAITE uses the ``[tool.pyright]`` section of its `pyproject.toml <https://github.com/mit-ll-ai-technology/maite/blob/main/pyproject.toml>`_ file 
to specify some Pyright configuration options, e.g., to exclude some folders from scans. 

See Also
--------
* `Static Typing with Python <https://typing.readthedocs.io/en/latest/>`_
* `A Primer on Python Typing: Relevant Language Features, Methods, and Tools for the T&E Framework <https://github.com/mit-ll-ai-technology/maite/blob/main/docs/source/explanation/type_hints_for_API_design.md>`_

