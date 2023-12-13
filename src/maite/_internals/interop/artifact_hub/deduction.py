# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import re
from typing import List, Pattern, get_args

from maite._internals.interop.provider import ArtifactName
from maite.errors import InvalidArgument
from maite.protocols import (
    Dataset,
    ImageClassifier,
    Metric,
    Model,
    ObjectDetectionDataset,
    ObjectDetector,
    TaskName,
    VisionDataset,
)


def _get_entrypoint_return_signature(entrypoint) -> bool:
    return inspect.signature(entrypoint).return_annotation


def identity(entrypoint):
    return True


def is_model_entrypoint(entrypoint) -> bool:
    return_t = _get_entrypoint_return_signature(entrypoint)
    return isinstance(return_t, Model)


def is_object_detector_entrypoint(entrypoint) -> bool:
    return_t = _get_entrypoint_return_signature(entrypoint)
    return isinstance(return_t, ObjectDetector)


def is_image_classifier_entrypoint(entrypoint) -> bool:
    return_t = _get_entrypoint_return_signature(entrypoint)
    return isinstance(return_t, ImageClassifier)


def is_dataset_entrypoint(entrypoint) -> bool:
    return_t = _get_entrypoint_return_signature(entrypoint)
    return isinstance(return_t, Dataset)


def is_vision_dataset_entrypoint(entrypoint) -> bool:
    return_t = _get_entrypoint_return_signature(entrypoint)
    return isinstance(return_t, VisionDataset)


def is_object_detection_dataset_entrypoint(entrypoint) -> bool:
    return_t = _get_entrypoint_return_signature(entrypoint)
    return isinstance(return_t, ObjectDetectionDataset)


def is_metric_entrypoint(entrypoint) -> bool:
    return_t = _get_entrypoint_return_signature(entrypoint)
    return isinstance(return_t, Metric)


def and_filters(*fs):
    return lambda ep: all(f(ep) for f in fs)


def or_filters(*fs):
    return lambda ep: any(f(ep) for f in fs)


def make_name_regex_filter(regex: re.Pattern | str):
    pat = regex if isinstance(regex, re.Pattern) else re.compile(regex)

    def name_matches_regex(entrypoint):
        return re.search(pat, entrypoint.__name__) is not None

    return name_matches_regex


def make_name_startswith_filter(prefix_str: str):
    def name_startswith_filter(entrypoint):
        return entrypoint.__name__.startswith(prefix_str)

    return name_startswith_filter


def make_name_match_filter(name: str):
    def name_match_filter(entrypoint):
        return entrypoint.__name__ == name

    return name_match_filter


def make_entrypoint_deduction_filter(
    require_deduction: bool = True,
    name: str | None = None,
    filter_type: ArtifactName | None = None,
    filter_task: TaskName | None = None,
    filter_regex: str | Pattern[str] | None = None,
    filter_str: str | List[str] | None = None,
):
    """Creates a filter for listed entrypoints on a module

    Parameters
    ----------
    require_deduction: bool, default True
        Require that the entrypoint has return annotations and they are runtime compatible with the Artifact Protocols
    name: str, default None
        Optionally require an exact match on name
    filter_type: ArtifactName | None
        model - entrypoint return implements the `Model` protocol
        dataset - entrypoint return implements the `Dataset` protocol
        metric - entrypoint return implements the `Metric` protocol
        None - entrypoint returns any of the above
    filter_task: 'image-classification' | 'object-detection' | None
        image-classification - entrypoint implements the specialization of the type protocol for image classification
        object-detection - entrypoint implements the specialization of the type protocol for object detection
        None: No additional restrictions placed on the entrypoint
    filter_str: str | List[str] , default None
        Optionally filter endpoints to those which begin with the name, or partial names provided


    """
    # Note: The composition of filters is very practical, but also redundant. If this
    # becomes a bottleneck we could make a smarter expression composition system which
    # unpacks and/or and simplifies. However, it is very likely that would be more
    # expensive than just double-checking some conditions as we are here. The
    # expressions are already built such that more general ones are evaluated first,
    # also `and` compositions will short-circuit.

    type_filters = {
        "model": is_model_entrypoint,
        "dataset": is_dataset_entrypoint,
        "metric": is_metric_entrypoint,
        None: or_filters(
            is_model_entrypoint, is_dataset_entrypoint, is_metric_entrypoint
        ),
    }

    task_filters = {
        "object-detection": or_filters(
            is_object_detection_dataset_entrypoint, is_object_detector_entrypoint
        ),
        # NOTE: the lack of parity in naming b/t dataset type and model type for OD above, and IC below
        "image-classification": or_filters(
            is_vision_dataset_entrypoint, is_image_classifier_entrypoint
        ),
        None: identity,
    }

    # Even without requiring deduction we can enforce that the entrypoint must be a callable
    # This is general enough, any symbol for which <symbol>(...) is a valid expression will pass.
    filter = callable

    if require_deduction:
        # if we require deduction, then lookup should return some filter for the
        # "<xyz>_filter" passed. this is the key, and both tables define that entry
        type_filter = type_filters.get(filter_type, None)
        if type_filter is None:
            raise InvalidArgument(
                f"Invalid type filter: {filter_type}, expected one of {get_args(ArtifactName)} or None"
            )

        task_filter = task_filters.get(filter_task, None)
        if task_filter is None:
            raise InvalidArgument(
                f"Invalid task filter: {filter_task} excepted one of {get_args(TaskName)} or None"
            )
        filter = and_filters(filter, task_filter, type_filter)

    else:
        if filter_type is not None or filter_task is not None:
            raise InvalidArgument(
                "Filtering on Artifact type or Task category is only possible with `require_deduction=True`"
            )

    if filter_regex is not None:
        filter = and_filters(filter, make_name_regex_filter(filter_regex))

    if filter_str is not None:
        strings = [filter_str] if not isinstance(filter_str, list) else filter_str
        name_filters = or_filters(*(make_name_startswith_filter(s) for s in strings))
        filter = and_filters(filter, name_filters)

    if name is not None:
        # this is the most restrictive condition that can be applied so putting it first makes sense as it will eagerly short circuit
        filter = and_filters(make_name_match_filter(name), filter)

    return filter
