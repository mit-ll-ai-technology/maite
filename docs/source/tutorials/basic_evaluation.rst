======================
Run a Basic Evaluation
======================

The JATIC Toolbox provides APIs for datasets, models, metrics, and evaluation
to make the use of JATIC capabilities more consistent across T&E tools.

In this tutorial, you will use methods from jatic-toolbox to:

- List datasets that are available from different providers, and load the CIFAR-10 test set from `TorchVision <https://pytorch.org/vision/stable/datasets.html>`__,
- List models that are available from different providers, and load a model from `HuggingFace <https://huggingface.co/datasets>`__ that has been pretrained on CIFAR-10,
- List metrics that are available from different providers, and load an accuracy metric from `TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`__, and
- Run an evaluation to compute the accuracy of the loaded model on the CIFAR-10 test set.

Once complete, you will have a basic understanding of the toolbox's APIs for loading data,
models, and metrics from various external libraries, and use the toolbox's native API for
running evaluations.

This tutorial does not assume any prior knowledge, but some experience with Python, machine learning,
and the PyTorch framework may be helpful.

Getting Started
===============

This tutorial requires PyTorch to be installed in your Python environment.
If you don't already have it installed, follow the instructions on the PyTorch website
`here <https://pytorch.org/get-started/locally/>`__ to install the version that matches
your OS, compute platform, and preferred package manager.

To install jatic-toolbox in your Python environment, run the following command in your terminal:

.. code:: shell

    pip install jatic-toolbox[all_interop]

``all_interop`` ensures you have the necessary packages to interoperate with external providers
such as TorchVision, HuggingFace, and TorchMetrics.
This will allow you to access datasets, models, and
metrics from these providers, which are needed for this tutorial.

Alternatively, you can also install the toolbox by cloning the git repository and installing locally:

.. code:: shell

    git clone git@gitlab.jatic.net:jatic/cdao/jatic-toolbox.git
    cd jatic-toolbox
    pip install .[all_interop]

To verify that the toolbox is installed as expected, open a Python console and try importing:

.. code:: pycon

    >>> import jatic_toolbox

We will now use jatic-toolbox to list and load a dataset, model, and metrics, and then use those to run
an evaluation.


Listing and Loading Datasets
============================

The JATIC toolbox provides APIs for listing and loading datasets from multiple providers,
including `TorchVision <https://pytorch.org/vision/stable/datasets.html>`__
and `HuggingFace <https://huggingface.co/datasets>`__.

List datasets
--------------

First, we'll import the ``list_datasets`` method,
and then use it to list the first 20 datasets that are available from TorchVision:

.. code:: pycon

    >>> from jatic_toolbox import list_datasets

.. code:: pycon

    >>> list_datasets(provider="torchvision")[:20]
    ['CIFAR10',
    'CIFAR100',
    'CLEVRClassification',
    'CREStereo',
    'Caltech101',
    'Caltech256',
    'CarlaStereo',
    'CelebA',
    'Cityscapes',
    'CocoCaptions',
    'CocoDetection',
    'Country211',
    'DTD',
    'DatasetFolder',
    'EMNIST',
    'ETH3DStereo',
    'EuroSAT',
    'FER2013',
    'FGVCAircraft',
    'FakeData']

Let's also count the number of datasets available from TorchVision, and compare it to the number of
datasets available from another provider, HuggingFace:

.. code:: pycon

    >>> len(list_datasets(provider="torchvision"))
    70
    >>> len(list_datasets(provider="huggingface"))
    751

Note that your numbers may differ slightly, as both providers are continuing to add new datasets.
We can also further filter the datasets from HuggingFace, such as only considering the datasets
for the task of image classification, and including datasets provided by the community:

.. code:: pycon

    >>> len(list_datasets(provider="huggingface", task_categories=["image-classification"]))
    13
    >>> len(list_datasets(provider="huggingface", task_categories=["image-classification"], with_community_datasets=True))
    231

Note that including community datasets provides us with a much larger number of potential datasets
to choose from.

Load a dataset
--------------

We'll use the CIFAR-10 dataset for this tutorial, due to its moderate size
and the availability of pretrained models.

To load the test set from the TorchVision version of CIFAR-10, we'll use the
toolbox's ``load_dataset`` method:

.. code:: pycon

    >>> from jatic_toolbox import load_dataset

.. code:: pycon

    >>> dataset = load_dataset(
    ...     provider="torchvision",
    ...     dataset_name="CIFAR10",
    ...     task="image-classification",
    ...     split="test",
    ...     root="~/data",
    ...     download=True
    ... )

Note that ``root`` indicates the directory where the dataset will be saved,
so feel free to change this value if you prefer to save your data in a different location.

Let's take a look at the first sample from the dataset:

.. code:: pycon

    >>> data = dataset[0]
    >>> print(data)
    {'image': <PIL.Image.Image image mode=RGB size=32x32 at 0x7FAD57236AC0>, 'label': 3}

Note that the data sample is in the form of a dictionary, with keys for ``image`` and ``label``.
The ``image`` is currently in the form of a PIL image.
For this tutorial, we'll be loading a model from the
`PyTorch <https://pytorch.org/>`__ framework,
so we'll need to convert the images in our dataset to
`Tensors <https://pytorch.org/docs/stable/tensors.html>`__.
To do this, we'll leverage TorchVision's ``to_tensor`` method:

.. code:: pycon

    >>> from torchvision.transforms.functional import to_tensor
    >>> dataset.set_transform(lambda x: {"image": to_tensor(x["image"]), "label": x["label"]})
    >>> data = dataset[0]
    >>> print(data["image"].shape)
    torch.Size([3, 32, 32])

Now that your dataset is configured, it's time to select and load a model to evaluate.


Listing and Loading Models
==========================

The JATIC Toolbox provides APIs for listing and loading models and pretrained weights from multiple providers,
including `TorchVision <https://pytorch.org/vision/stable/models.html>`__
and `HuggingFace <https://huggingface.co/models>`__.

Listing Models
--------------

We'll start by using the ``list_models`` method to explore the models that are
available from TorchVision and HuggingFace for image classification:

.. code:: pycon

    >>> from jatic_toolbox import list_models

.. code:: pycon

    >>> len(list_models(provider="torchvision", task="image-classification"))
    80
    >>> len(list_models(provider="huggingface", task="image-classification"))
    3523

Since we will be testing on the CIFAR-10 dataset, we'd like to load a model that we
know has been pretrained on CIFAR-10.
We can find these candidate models from HuggingFace by searching for models that
contain "cifar10" in their ``model_name``.
Let's look at the first 20 models in this list:

.. code:: pycon

    >>> models = list_models(provider="huggingface", task="image-classification", model_name="cifar10")
    >>> sorted([m.id for m in models])[:20]
    ['Ahmed9275/Vit-Cifar100',
    'LaCarnevali/vit-cifar10',
    'MazenAmria/swin-base-finetuned-cifar100',
    'MazenAmria/swin-small-finetuned-cifar100',
    'MazenAmria/swin-tiny-finetuned-cifar100',
    'SajjadAlam/beit_Cifar10_finetune_model',
    'Weili/resnet-18-cifar100',
    'Weili/swin-base-patch4-window7-224-in22k-finetuned-cifar10',
    'Weili/swin-tiny-patch4-window7-224-finetuned-cifar10',
    'Weili/vit-base-patch16-224-finetuned-cifar10',
    'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',
    'abhinavkk/cifar10_model',
    'abhishek/autotrain_cifar10_vit_base',
    'ahsanjavid/convnext-tiny-finetuned-cifar10',
    'alfredcs/swin-cifar10',
    'alfredcs/vit-cifar10',
    'amehta633/cifar-10-vgg-pretrained',
    'arize-ai/resnet-50-cifar10-quality-drift',
    'edadaltocg/densenet121_cifar10',
    'edadaltocg/densenet121_cifar100']

Note that your list may look slightly different, since new models are constantly being added to
the HuggingFace hub by users in the community.

Load a Model
------------

Next, let's load one of those HuggingFace models using the toolbox's ``load_model`` method:

.. code:: pycon

    >>> from jatic_toolbox import load_model

.. code:: pycon

    >>> model = load_model(
    ...     provider="huggingface",
    ...     model_name="ahsanjavid/convnext-tiny-finetuned-cifar10",
    ...     task="image-classification"
    ... )

Verify that you can pass an input from your dataset through this model to
get a prediction:

.. code:: pycon

    >>> data = model.preprocessor([dataset[0]])
    >>> input = data[0]["image"].unsqueeze(0)
    >>> output = model(input)

Finally, compare the model's prediction (i.e., output with highest logit value) to truth:

.. code:: pycon

    >>> print(output.logits.argmax(dim=1).item())
    3
    >>> print(dataset[0]["label"])
    3

Notice that for this model, the prediction matches truth! However, if you loaded a 
different model (or a different dataset), you may end up with a different outcome.

Now that we've verified we can run a single input through our model to get an output,
let's load a metric that we'll use to compute performance of the model across the entire dataset.


Listing and Loading Metrics
===========================

The JATIC Toolbox also provides APIs for listing and loading metrics from common providers,
including `TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`__
and `TorchEval <https://github.com/pytorch/torcheval>`__.

Listing Metrics
---------------

We'll start by using the ``list_metrics`` method to compare the number of metrics from each provider:

.. code:: pycon

    >>> from jatic_toolbox import list_metrics

.. code:: pycon

    >>> len(list_metrics(provider="torchmetrics"))
    79
    >>> len(list_metrics(provider="torcheval"))
    50

Let's list the first 20 metrics from each provider:

.. code:: pycon

    >>> list_metrics(provider="torchmetrics")[:20]
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

.. code:: pycon

    >>> list_metrics(provider="torcheval")[:20]
    ['AUC',
    'BinaryAccuracy',
    'BinaryAUPRC',
    'BinaryAUROC',
    'BinaryBinnedAUROC',
    'BinaryBinnedPrecisionRecallCurve',
    'BinaryConfusionMatrix',
    'BinaryF1Score',
    'BinaryNormalizedEntropy',
    'BinaryPrecision',
    'BinaryPrecisionRecallCurve',
    'BinaryRecall',
    'BinaryRecallAtFixedPrecision',
    'BLEUScore',
    'Cat',
    'ClickThroughRate',
    'HitRate',
    'Max',
    'Mean',
    'MeanSquaredError']

Note that each provider has similar metrics, but often with different names (e.g., AUROC vs. AUC).

Load Metrics
------------

For this tutorial, we'll evaluate the performance of our model using a common metric: accuracy.
We'll use the toolbox's ``load_metric`` method to load and configure the accuracy metric from TorchMetrics:

.. code:: pycon

    >>> from jatic_toolbox import load_metric

.. code:: pycon

    >>> metric = load_metric(provider="torchmetrics", metric_name="Accuracy", task="multiclass", num_classes=10)

You are now ready to run a full evaluation using your dataset, model, and metric!

Run an Evaluation
=================

First, instantiate an evaluator using the jatic-toolbox ``evaluate`` method:

.. code:: pycon

    >>> from jatic_toolbox import evaluate

.. code:: pycon

    >>> evaluator = evaluate(task="image-classification")

Next, run the evaluator to compute metrics using your dataset and model:

.. code:: pycon

    >>> output = evaluator(
    ...     model,
    ...     dataset,
    ...     metric=dict(accuracy=metric),
    ...     batch_size=32,
    ... )

Note this may take a while since it is iterating through the full test dataset.
Once complete, print your results:

.. code:: pycon

    >>> print(output)
    {'accuracy': tensor(0.9736)}

Congrats! You have now successfully used the jatic-toolbox to load a dataset, model,
and metric from external providers, and run an evaluation to compute the accuracy of
the loaded model on the CIFAR-10 test dataset.
