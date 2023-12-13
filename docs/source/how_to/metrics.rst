=================================
List, Load, and Configure Metrics
=================================

The MAITE defines a ``Metric`` protocol that standardizes
the metric interface across several very similar libraries
(e.g., `TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`__,
`TorchEval <https://github.com/pytorch/torcheval>`__)
with common methods:

- ``reset``
- ``update``
- ``compute``, and
- ``to(...)``

The maite also provides APIs for listing and loading metrics from common providers,
including `TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`__ and
`TorchEval <https://github.com/pytorch/torcheval>`__,
which adhere to this protocol.

In this How-To, we will walk through the steps of:

1. Listing available metrics from a provider
2. Loading a metric from a provider
3. Verifying that the metric conforms to the maite ``Metric`` protocol
4. Loading and configuring multiple metrics for use with maite's ``evaluate``

0. Installation
===============
If you haven't already, make sure that you've installed the library for the
metrics provider that you are planning to use (e.g., ``pip install torchmetrics``).

If you want to install all libraries associated with the maite's interoperability
capabilities, you can install them along with the maite by running:

.. code:: console

   $ pip install maite[all_interop]

1. List available metrics from a provider
=========================================
Use the ``list_metrics`` method to list available metrics from one of the maite's
supported providers:

.. code:: pycon

    >>> from maite import list_metrics
    >>> print(list_metrics(provider="torchmetrics")[:20])
    ['Accuracy',
    'AUROC',
    'AveragePrecision',
    'BLEUScore',
    'CalibrationError',
    'CatMetric',
    'CharErrorRate',
    'CHRFScore',
    'ConcordanceCorrCoef',
    'CohenKappa',
    'ConfusionMatrix',
    'CosineSimilarity',
    'CramersV',
    'Dice',
    'TweedieDevianceScore',
    'ErrorRelativeGlobalDimensionlessSynthesis',
    'ExactMatch',
    'ExplainedVariance',
    'ExtendedEditDistance',
    'F1Score']

2. Load a metric from a provider
================================
Once you've identified the provider and metric you would like to use, load the metric using the
``load_metric`` method by specifying the ``provider``, ``metric_name``, and any
additional arguments needed to configure that metric.

Here is an example of loading the accuracy metric from TorchMetrics:

.. code:: python

    from maite import load_metric
    
    metric = load_metric(
        provider="torchmetrics",
        metric_name="Accuracy",
        task="multiclass",
        num_classes=4,
        average="micro",
    )

The keyword arguments (e.g., ``task``, ``num_classes``, ``average``)
are specific to the provider and metric you are loading,
so consult the documentation from the provider for more information on what parameters
are needed to configure a specific metric.
Also, make sure to assign values for these parameters appropriately for your application
(e.g., if you are testing a model trained on the CIFAR-10 dataset, which contains 10 classes,
set ``num_classes=10``).

Here is an example of using the metric on some toy data:

.. code:: pycon

    >>> import torch
    >>> target = torch.tensor([0, 1, 2, 3])
    >>> preds = torch.tensor([0, 2, 1, 3])
    >>> metric(preds, target)
    tensor(0.5000)

3. Verify that the metric conforms to the maite ``Metric`` protocol
===========================================================================
Verify that the metric you just loaded conforms to the maite's protocol
by asserting that the loaded metric is an instance of ``Metric``:

.. code:: pycon

    >>> from maite.protocols import Metric
    >>> assert isinstance(metric, Metric)

This assertion should pass.

You can also verify that the class methods (e.g., ``reset``, ``update``, ``compute``)
are working properly using some toy data:

.. code:: pycon

    >>> target = torch.tensor([0, 1, 2, 3])
    >>> preds1 = torch.tensor([0, 2, 1, 3])
    >>> preds2 = torch.tensor([0, 1, 2, 3])

    >>> metric.reset()
    >>> metric.update(preds1, target)
    >>> print(metric.compute())
    tensor(0.5000)

    >>> metric.update(preds2, target)
    >>> print(metric.compute())
    tensor(0.7500)

    >>> metric.reset()
    >>> metric.update(preds2, target)
    >>> print(metric.compute())
    tensor(1.)

Note that the second time ``metric.compute()`` is called,
the output reflects the accuracy using both ``preds1`` and ``preds2``.
After calling ``metric.reset()`` the second time,
the accuracy is only computed using ``preds2``.

4. Load and configure multiple metrics for use with maite's ``evaluate``
================================================================================
If you would like to evaluate a model and dataset against multiple metrics at the same time,
a collection of metrics can be loaded and saved as a dictionary,
and then passed into the maite's ``evaluate`` method.

Here is an example of configuring two different metrics:

.. code:: python

    from maite import load_metric

    metrics = dict(
        accuracy=load_metric(
            provider="torchmetrics",
            metric_name="Accuracy",
            task="multiclass",
            num_classes=10,
            average="none",
        ),
        f1score=load_metric(
            provider="torchmetrics",
            metric_name="F1Score",
            task="multiclass",
            num_classes=10,
            average="none",
        )
    )

Note that while in this example we loaded all of the metrics from the
TorchMetrics provider, the maite's standard APIs also allow for
mix and matching metrics from multiple providers.

The metrics can now be passed to an instantiation of the maite's ``evaluate`` method
to run an evaluation given a previously loaded ``model`` and ``dataset``:

.. code:: python
    
    from maite import evaluate

    evaluator = evaluate(task="image-classification")

    # Reset metrics before running evaluation
    [m.reset() for m in metrics.values()]

    output = evaluator(
        model,
        dataset,
        metric=metrics,
        batch_size=32,
    )

You can then access your metrics from the dictionary output of the evaluator:

.. code:: pycon

    >>> print(output)
    {'accuracy': tensor([0.9910, 0.9880, 0.9590, 0.9420, 0.9840, 0.9420, 0.9940, 0.9810, 0.9840, 0.9710]),
    'f1score': tensor([0.9783, 0.9826, 0.9726, 0.9401, 0.9757, 0.9515, 0.9851, 0.9854, 0.9865, 0.9778])}

Note that these values will vary depending on the dataset, model, and metrics you are
using to run your evaluation.
In this example, each metric output contains 10 values because we are evaluating against a dataset with 10 classes,
and set ``num_classes=10`` and ``average="none"`` when configuring our metrics.
This configuration allowed us to compute the metrics for each of the 10 classes individually.

We have now walked through the process of listing, loading, and configuring a collection of
metrics from an external provider (e.g., TorchMetrics) for use in an evaluation,
as well as verifying a loaded metric adheres to the maite's ``Metric`` protocol.

It is also possible to define custom metrics that conform to the ``Metric`` protocol
and can be used with ``evaluate``,
but these steps will be reserved for a future How-To.
