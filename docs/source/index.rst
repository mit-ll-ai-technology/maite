.. maite documentation master file, created by
   sphinx-quickstart on Fri Apr 23 17:23:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=========================================
Welcome to MAITE's documentation!
=========================================

The Modular AI Trustworthy Engineering (MAITE) library provides common interfaces, 
utilities, and tooling to streamline the development of test and evaluation (T&E) capabilities, 
support core evaluation use cases, and allow composition of T&E tools into custom workflows.

Installation
============

The core of MAITE is lightweight: its only dependencies are ``typing-extensions`` and ``numpy``. 
To install it, run:

.. code:: console

   $ pip install maite

If instead you want to try out the features in the upcoming version, you can install 
the latest pre-release of MAITE with:

.. code:: console

   $ pip install --pre maite

Learning about MAITE
============================

Our docs are divided into four sections: Tutorials, How-Tos, Explanations, and 
Reference.

If you want to get a bird's-eye view of what MAITE is all about check out our 
**Tutorials**. For folks who have already used the MAITE library, our **How-Tos** and 
**Reference** materials can help acquaint you with the unique capabilities that are 
offered by MAITE. Finally, **Explanations** provide readers with taxonomies, 
design principles, recommendations, and other articles that will enrich their 
understanding of the MAITE library.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   how_tos
   explanation
   api_reference
   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
