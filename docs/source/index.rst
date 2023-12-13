.. maite documentation master file, created by
   sphinx-quickstart on Fri Apr 23 17:23:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=========================================
Welcome to maite's documentation!
=========================================


Installation
============

The core of maite is lightweight: its only dependencies is ``typing-extensions``. 
To install it, run:

.. code:: console

   $ pip install maite

If instead you want to try out the features in the upcoming version, you can install 
the latest pre-release of maite with:

.. code:: console

   $ pip install --pre maite

The toolbox also contains APIs and helper utilities to provide interoperability with
various other open-source libraries such as
`HuggingFace <https://huggingface.co/datasets>`_,
`TorchVision <https://pytorch.org/vision/stable/datasets.html>`_, and
`TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`_.
To leverage these capabilities, run the following command to install maite
along with these other dependencies:

.. code:: console

   $ pip install maite[all_interop]


Learning About maite
============================

Our docs are divided into four sections: Tutorials, How-Tos, Explanations, and 
Reference.

If you want to get a bird's-eye view of what maite is all about check out our 
**Tutorials**. For folks who have used the maite, our **How-Tos** and 
**Reference** materials can help acquaint you with the unique capabilities that are 
offered by maite. Finally, **Explanations** provide readers with taxonomies, 
design principles, recommendations, and other articles that will enrich their 
understanding of maite.

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
