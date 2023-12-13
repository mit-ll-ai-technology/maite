.. meta::
   :description: How-To use registered models, datasets, and metrics.


============================================
Use Registered Models, Datasets, and Metrics
============================================

The MAITE includes a list of of registered models, datasets, and metrics 
that can be found at the top level of the MAITE. 

.. code-block:: python

    from maite import MODEL_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY

    print(MODEL_REGISTRY.keys())
    print(DATASET_REGISTRY.keys())
    print(METRIC_REGISTRY.keys())

These registries are provided at the top level to allow developers and users to easily
update or add new models, datasets, and metrics.

These registries can be used to load models, datasets, and metrics with default values. 
Supported keyword arguments will override the default values.

.. code-block:: python

    import maite
     
    model = maite.load_model(model_name="vit_for_cifar10")
     
    # override default split of "test"
    dataset = maite.load_dataset(dataset_name="cifar10-test", split="test[:100]")
    
    # must set num_classes
    metric = maite.load_metric(metric_name="classification_report", num_classes=10)


These registries can also be used to run :class:`maite.evaluate` with named arguments.
To override default values, pass a dictionary of keyword arguments to ```model_kwargs``,
``dataset_kwargs`` and ``metric_kwargs``. 

.. code-block:: python

    import maite

    evaluator = maite.evaluate("image-classification")
    output = evaluator(
        model="vit_for_cifar10",
        data="cifar10-test",
        metric="classification_report",
        batch_size=32,
        device=0,
        dataset_kwargs=dict(plit="test[:100]"),
        metric_kwargs=dict(num_classes=10)
    )

    report = {k: v.detach().cpu().numpy() for k, v  in output["classification_report"].items()}

    import pandas as pd
    pd.DataFrame.from_dict(report)

.. code-block:: console

        accuracy	f1score	precision	recall
    0	1.000000	1.000000	1.000000	1.000000
    1	0.833333	0.909091	1.000000	0.833333
    2	1.000000	1.000000	1.000000	1.000000
    3	1.000000	1.000000	1.000000	1.000000
    4	1.000000	1.000000	1.000000	1.000000
    5	1.000000	1.000000	1.000000	1.000000
    6	1.000000	1.000000	1.000000	1.000000
    7	1.000000	1.000000	1.000000	1.000000
    8	1.000000	0.962963	0.928571	1.000000
    9	0.909091	0.909091	0.909091	0.909091