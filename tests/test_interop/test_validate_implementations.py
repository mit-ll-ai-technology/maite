import pytest

from jatic_toolbox._internals.import_utils import is_hf_hub_available
from jatic_toolbox.testing.pyright import pyright_analyze


@pytest.mark.skipif(not is_hf_hub_available(), reason="HuggingFace is not installed.")
def test_huggingface_image_classifier():
    def func():
        import typing as tp
        from dataclasses import dataclass

        import torch as tr

        from jatic_toolbox import protocols as pr
        from jatic_toolbox._internals.interop.huggingface.typing import BatchFeature
        from jatic_toolbox.interop.huggingface import HuggingFaceImageClassifier

        @dataclass
        class Meta:
            id2label: dict
            num_labels: int

        def get_test_vision_model():
            @dataclass
            class VisionOutput:
                logits: tp.Union[pr.ArrayLike, tp.Sequence[pr.ArrayLike]]

            class Processor:
                def __call__(
                    self,
                    images: tp.Sequence[pr.ArrayLike],
                    return_tensors: tp.Union[bool, str] = "pt",
                    **kwargs: tp.Any,
                ):
                    images = [tr.as_tensor(i) for i in images]
                    return BatchFeature(pixel_values=tr.stack(images))

            class Model(tr.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.device = "cpu"
                    self.config = Meta(
                        id2label={i: f"label_{i}" for i in range(10)}, num_labels=10
                    )

                def forward(self, *args, **kwargs) -> pr.HasLogits:
                    logits = tr.randn(1, 10)
                    return VisionOutput(logits)

            return Processor(), Model()

        def f(model: pr.ImageClassifier):
            ...

        processor, model = get_test_vision_model()
        f(HuggingFaceImageClassifier(model, processor))

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 0


@pytest.mark.skipif(not is_hf_hub_available(), reason="HuggingFace is not installed.")
def test_huggingface_object_detector():
    def func():
        import typing as tp
        from dataclasses import dataclass

        import torch as tr
        from transformers.utils import ModelOutput

        from jatic_toolbox import protocols as pr
        from jatic_toolbox._internals.interop.huggingface.typing import BatchFeature
        from jatic_toolbox.interop.huggingface import HuggingFaceObjectDetector

        @dataclass
        class Meta:
            id2label: dict
            num_labels: int

        @dataclass
        class ObjectDetectionWithLogits(ModelOutput):
            logits: pr.ArrayLike = None  # type: ignore[assignment]
            pred_boxes: pr.ArrayLike = None  # type: ignore[assignment]

        @dataclass
        class ObjectDetectionOutput(ModelOutput):
            boxes: tr.Tensor = None  # type: ignore[assignment]
            scores: tr.Tensor = None  # type: ignore[assignment]
            labels: tr.Tensor = None  # type: ignore[assignment]

        def get_test_object_detection_model(output_as_list=False):
            class Processor:
                def __call__(
                    self,
                    images: tp.Sequence[pr.ArrayLike],
                    return_tensors: tp.Union[bool, str] = "pt",
                    **kwargs: tp.Any,
                ):
                    images = [tr.as_tensor(i) for i in images]
                    return BatchFeature(pixel_values=tr.stack(images))

                def post_process_object_detection(
                    self, outputs, threshold, target_sizes
                ):
                    boxes = tr.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
                    scores = tr.tensor([0.5, 0.5])
                    labels = tr.tensor([0, 1])
                    if output_as_list:
                        return [
                            dict(boxes=b, scores=s, labels=l)
                            for b, s, l in zip(boxes, scores, labels)
                        ]
                    return ObjectDetectionOutput(boxes, scores, labels)

            class Model(tr.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.device = "cpu"
                    self.config = Meta(
                        id2label={i: f"label_{i}" for i in range(10)}, num_labels=10
                    )

                def forward(self, *args, **kwargs):
                    boxes = tr.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
                    logits = tr.randn(1, 10)
                    return ObjectDetectionWithLogits(logits, boxes)

            return Processor(), Model()

        def f(model: pr.ObjectDetector):
            ...

        processor, model = get_test_object_detection_model()
        f(HuggingFaceObjectDetector(model, processor))

    x = pyright_analyze(func)
    print(x["generalDiagnostics"])
    assert x["summary"]["errorCount"] == 0, x["generalDiagnostics"]
