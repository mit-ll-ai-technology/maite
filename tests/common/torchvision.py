import torch as tr
from torch.utils.data import Dataset

from .utils import create_random_image


def get_test_vision_dataset(has_split, has_train):
    """
    Creates a test dataset for testing vision datasets.
    """

    class TVDataset(Dataset):
        def __init__(self, *, split=None, train=None):
            if not has_split and split is not None:
                raise TypeError("split is not supported")

            if not has_train and train is not None:
                raise TypeError("train is not supported")

            self.classes = ["a", "b", "c", "d"]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            image = create_random_image()
            label = 3
            return image, label

    return TVDataset


def get_test_vision_model():
    from torchvision.models import WeightsEnum

    class Transform:
        def __call__(self, x):
            return x

    class Weights:
        def __init__(self):
            self.weights = tr.randn(1, 3, 10, 10)

        def transforms(self):
            return Transform()

    class TVWeights(WeightsEnum):
        DEFAULT = Weights()

    class TVModel(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = tr.nn.Linear(1, 1)

        def forward(self, x):
            return x

    return TVWeights, TVModel()


def get_test_object_detection_model():
    from torchvision.models import WeightsEnum

    class Transform:
        def __call__(self, x):
            return x

    class Weights:
        def __init__(self):
            self.weights = tr.randn(1, 3, 10, 10)

        def transforms(self):
            return Transform()

    class TVWeights(WeightsEnum):
        DEFAULT = Weights()

    class TVModel(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = tr.nn.Linear(1, 1)

        def forward(self, x):
            boxes = tr.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
            scores = tr.tensor([0.5, 0.5])
            labels = tr.tensor([0, 1])
            return [
                dict(boxes=b, scores=s, labels=l)
                for b, s, l in zip(boxes, scores, labels)
            ]

    return TVWeights, TVModel()
