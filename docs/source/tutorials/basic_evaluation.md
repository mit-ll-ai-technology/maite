> Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY<br/>
> Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).<br/>
> SPDX-License-Identifier: MIT

# Run a Basic Evaluation

The MAITE library provides APIs for datasets, models, metrics, and evaluation to make their use more consistent across test and evaluation (T&E) tools.

In this tutorial, you will use methods from MAITE to:

* List datasets that are available from different providers, and load the CIFAR-10 test set from [HuggingFace](https://huggingface.co/datasets),
* List models that are available from different providers, and load a model from [HuggingFace](https://huggingface.co/models) that has been pretrained on CIFAR-10,
* List metrics that are available from different providers, and load an accuracy metric from [TorchMetrics](https://github.com/Lightning-AI/torchmetrics), and
* Run an evaluation to compute the accuracy of the loaded model on the CIFAR-10 test set.

Once complete, you will have a basic understanding of MAITE’s APIs for loading datasets, models, and metrics from various external libraries, and how to use MAITE’s native API for running evaluations.

This tutorial does not assume any prior knowledge, but some experience with Python, machine learning, and the PyTorch framework may be helpful.

Note: This tutorial can be found as a Jupyter notebook [here](https://github.com/mit-ll-ai-technology/maite/examples/basic_evaluation.ipynb).

## Getting Started
This tutorial requires PyTorch to be installed in your Python environment. If you don’t already have it installed, follow the instructions on the PyTorch website [here](https://pytorch.org/get-started/locally/) to install the version that matches your OS, compute platform, and preferred package manager.

To install MAITE in your Python environment, run the following command in your terminal:

```python
pip install maite[all_interop]
```

``all_interop`` ensures you have the necessary packages to interoperate with external providers such as TorchVision, HuggingFace, and TorchMetrics. This will allow you to access datasets, models, and metrics from these providers, which are needed for this tutorial.

Alternatively, you can also install MAITE by cloning the git repository and installing locally:
```python
git clone git@gitlab.jatic.net:jatic/cdao/maite.git
cd maite
pip install .[all_interop]
```

To verify that MAITE is installed as expected, open a Python console and try importing:
```
>>> import maite
```
We will now use MAITE to list and load a dataset, model, and metrics, and then use those to run an evaluation.

## Listing and Loading Datasets
The MAITE library provides APIs for listing and loading datasets from multiple providers, including [TorchVision](https://pytorch.org/vision/stable/datasets.html) and [HuggingFace](https://huggingface.co/datasets).

### List datasets
First, we’ll import the ``list_datasets`` method, and then use it to list the first 20 datasets that are available from TorchVision:


```python
from maite import list_datasets

list_datasets(provider="torchvision")[:20]
```




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



Let’s also count the number of datasets available from TorchVision, and compare it to the number of datasets available from another provider, HuggingFace:


```python
len(list_datasets(provider="torchvision"))
```




    71




```python
len(list_datasets(provider="huggingface"))
```




    740



Note that your numbers may differ slightly, as both providers are continuing to add new datasets. We can also further filter the datasets from HuggingFace, such as only considering the datasets for the task of image classification, and including datasets provided by the community:


```python
len(list_datasets(
    provider="huggingface",
    task_categories=["image-classification"]
))
```




    12




```python
len(list(list_datasets(
    provider="huggingface",
    task_categories=["image-classification"],
    with_community_datasets=True
)))
```




    669



Note that including community datasets provides us with a much larger number of potential datasets to choose from.

## Load a dataset
We’ll use the CIFAR-10 dataset for this tutorial, due to its moderate size and the availability of pretrained models.

To load the test set from the HuggingFace version of CIFAR-10, we’ll use MAITE’s ``load_dataset`` method:


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
from maite import load_dataset

dataset = load_dataset(
    provider="huggingface",
    dataset_name="cifar10",
    task="image-classification",
    split="test"
)
```

Let’s take a look at the first sample from the dataset:


```python
data = dataset[0]
print(data)
```

    {'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x2ABB8DF0D90>, 'label': 3}
    

Note that the data sample is in the form of a dictionary, with keys for ``image`` and ``label``. The ``image`` is currently in the form of a PIL image. For this tutorial, we’ll be loading a model from the [PyTorch](https://pytorch.org/) framework, so we’ll need to convert the images in our dataset to [Tensors](https://pytorch.org/docs/stable/tensors.html). To do this, we’ll leverage TorchVision’s ``to_tensor`` method, with a resize to match our eventual model:


```python
from torchvision.transforms.functional import to_tensor
dataset.set_transform(
    lambda x: {
        "image": to_tensor(x["image"].resize((224,224))),
        "label": x["label"]
    }
)
data = dataset[0]
print(data["image"].shape)
```

    torch.Size([3, 224, 224])
    

Now that your dataset is configured, it’s time to select and load a model to evaluate.

## Listing and Loading Models
The MAITE library  provides APIs for listing and loading models and pretrained weights from multiple providers, including [TorchVision](https://pytorch.org/vision/stable/models.html) and [HuggingFace](https://huggingface.co/models).

### Listing Models
We’ll start by using the ``list_models`` method to explore the models that are available from TorchVision and HuggingFace for image classification:


```python
from maite import list_models

len(list_models(provider="torchvision", task="image-classification"))
```




    80




```python
len(list_models(provider="huggingface", task="image-classification"))
```




    8168



Since we will be testing on the CIFAR-10 dataset, we’d like to load a model that we know has been pretrained on CIFAR-10. We can find these candidate models from HuggingFace by searching for models that contain “cifar10” in their ``model_name``. Let’s look at the first 20 models in this list:


```python
models = list_models(
    provider="huggingface",
    task="image-classification",
    model_name="cifar10"
)
sorted([m.id for m in models])[:20]
```




    ['02shanky/vit-finetuned-cifar10',
     '02shanky/vit-finetuned-vanilla-cifar10-0',
     'Ahmed9275/Vit-Cifar100',
     'DeepCyber/Enhanced-CIFAR10-CNN',
     'JamesCS462/JamesCS462_cifar100',
     'LaCarnevali/vit-cifar10',
     'MazenAmria/swin-base-finetuned-cifar100',
     'MazenAmria/swin-small-finetuned-cifar100',
     'MazenAmria/swin-tiny-finetuned-cifar100',
     'NouRed/fine-tuned-vit-cifar10',
     'SajjadAlam/beit_Cifar10_finetune_model',
     'Sendeky/Cifar10',
     'Skafu/swin-tiny-patch4-window7-224-cifar10',
     'Skafu/swin-tiny-patch4-window7-224-finetuned-eurosat-finetuned-cifar100',
     'TirathP/cifar10-lt',
     'Weili/resnet-18-cifar100',
     'Weili/swin-base-patch4-window7-224-in22k-finetuned-cifar10',
     'Weili/swin-tiny-patch4-window7-224-finetuned-cifar10',
     'Weili/vit-base-patch16-224-finetuned-cifar10',
     'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10']



Note that your list may look slightly different, since new models are constantly being added to the HuggingFace hub by users in the community.

## Load a Model
Next, let’s load one of those HuggingFace models using MAITE’s ``load_model`` method:


```python
from maite import load_model

model = load_model(
    provider="huggingface",
    model_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    task="image-classification"
)
```

Verify that you can pass an input from your dataset through this model to get a prediction:


```python
data = dataset[0]
output = model(data["image"])
```

Finally, compare the model’s prediction (i.e., output with highest probability) to truth:


```python
print(output.probs.argmax(dim=1).item())
```

    3
    


```python
print(dataset[0]["label"])
```

    3
    

Notice that for this model, the prediction matches truth! However, if you loaded a different model (or a different dataset), you may end up with a different outcome.

Now that we’ve verified we can run a single input through our model to get an output, let’s load a metric that we’ll use to compute performance of the model across the entire dataset.

## Listing and Loading Metrics
The MAITE library also provides APIs for listing and loading metrics from common providers, including [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) and [TorchEval](https://github.com/pytorch/torcheval).

### Listing Metrics
We’ll start by using the ``list_metrics`` method to compare the number of metrics from each provider:


```python
from maite import list_metrics

len(list_metrics(provider="torchmetrics"))
```




    79




```python
len(list_metrics(provider="torcheval"))
```




    57



Let’s list the first 20 metrics from each provider:


```python
list_metrics(provider="torchmetrics")[:20]
```




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




```python
list_metrics(provider="torcheval")[:20]
```




    ['AUC',
     'BinaryAccuracy',
     'BinaryAUPRC',
     'BinaryAUROC',
     'BinaryBinnedAUPRC',
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
     'FrechetInceptionDistance',
     'HitRate',
     'Max']



Note that each provider has similar metrics, but often with different names (e.g., AUROC vs. AUC).

### Load Metrics
For this tutorial, we’ll evaluate the performance of our model using a common metric: accuracy. We’ll use the MAITE ``load_metric`` method to load and configure the accuracy metric from TorchMetrics:


```python
from maite import load_metric

metric = load_metric(
    provider="torchmetrics",
    metric_name="Accuracy",
    task="multiclass",
    num_classes=10
)
```

You are now ready to run a full evaluation using your dataset, model, and metric!

## Run an Evaluation
First, instantiate an evaluator using the MAITE ``evaluate`` method:


```python
from maite import evaluate

evaluator = evaluate(task="image-classification")
```

Next, run the evaluator to compute metrics using a subset of your dataset and model:


```python
dataset_subset = [dataset[i] for i in range(256)]
```


```python
output = evaluator(
    model,
    dataset_subset,
    metric=dict(accuracy=metric),
    batch_size=32,
)
```


      0%|          | 0/8 [00:00<?, ?it/s]


Once complete, print your results:


```python
print(output)
```

    {'accuracy': tensor(0.9531)}
    

Congrats! You have now successfully used MAITE to load a dataset, model, and metric from external providers, and run an evaluation to compute the accuracy of the loaded model on the CIFAR-10 test dataset.
