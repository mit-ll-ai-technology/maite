"""Concrete tests of spotcheck API ability to detect 'bad' versions of MAITE components
when used to decorate MAITE tasks that type arguments/return values as such

We systematically iterate dummy component implementers, selectively corrupt them,
so they break some predicate, and run through a spotcheck-decorated `evaluate` to confirm
that an exception is raised.
"""

import importlib
from typing import Literal

import pytest

from maite._internals.compat import TypedDict
from maite.protocols import generic as gen
from maite.spotcheck import SpotcheckError, spotcheck
from maite.tasks import evaluate

COMPONENT_PROTOCOL_NAMES = ["Dataset", "DataLoader", "Augmentation", "Model"]

AI_PROBLEM_NAMES = ["image_classification", "object_detection", "multiobject_tracking"]

CORE_PROTOCOL_MODULE_PREFIX = "maite._internals.protocols"
DUMMY_PROTOCOL_IMPL_MODULE_PREFIX = "maite._internals.dummy_protocols"


class ComponentDict(TypedDict):
    dataset: gen.Dataset
    dataloader: gen.DataLoader
    augmentation: gen.Augmentation
    model: gen.Model
    metric: gen.Metric


def get_good_impl_set(
    ai_problem_name: Literal["object_detection", "image_classification", "multiobject_tracking"],
) -> ComponentDict:
    """Retrieve a dictionary containing statically/dynamically valid components"""

    impl_module = importlib.import_module(DUMMY_PROTOCOL_IMPL_MODULE_PREFIX + "." + ai_problem_name)
    return {
        "dataset": impl_module.Dataset(),
        "dataloader": impl_module.DataLoader(),
        "augmentation": impl_module.Augmentation(),
        "model": impl_module.Model(),
        "metric": impl_module.Metric(),
    }


def get_bad_impl(
    ai_problem_name: Literal["object_detection", "image_classification", "multiobject_tracking"],
    component_protocol_name: Literal["Dataset", "DataLoader", "Augmentation", "Model", "Metric"],
) -> tuple[str, object]:
    """Retrieve a 'bad' implementation of some component protocol with a given AI problem"""
    # Need to ensure that we return with *one* bad protocol implementer
    # based on component_protocol_name and otherwise all good ones

    impl_module = importlib.import_module(DUMMY_PROTOCOL_IMPL_MODULE_PREFIX + "." + ai_problem_name)
    badimpl_cls = getattr(impl_module, "Bad" + component_protocol_name)
    return component_protocol_name.lower(), badimpl_cls()


@pytest.mark.parametrize("ai_problem_name", AI_PROBLEM_NAMES)
@pytest.mark.parametrize("component_protocol_name", COMPONENT_PROTOCOL_NAMES)
def test_good_bad_evaluate_call(ai_problem_name, component_protocol_name):

    checked_evaluate = spotcheck(evaluate, ai_problem=ai_problem_name)

    impl_dict = get_good_impl_set(ai_problem_name)

    # call evaluate with both only dataset and only dataloader
    # (should run normally and not be stopped by 'spotcheck')
    _ = checked_evaluate(
        dataset=impl_dict["dataset"],
        model=impl_dict["model"],
        augmentation=impl_dict["augmentation"],
        metric=impl_dict["metric"],
    )

    _ = checked_evaluate(
        dataloader=impl_dict["dataloader"],
        model=impl_dict["model"],
        augmentation=impl_dict["augmentation"],
        metric=impl_dict["metric"],
    )

    bad_impl_protocol_name, bad_impl = get_bad_impl(ai_problem_name, component_protocol_name)

    # spotchecked evaluate passes with impl_cls/proto_cls
    impl_dict[bad_impl_protocol_name] = bad_impl

    # call checked_evaluate with both only dataset and only dataloader
    if bad_impl_protocol_name != "dataloader":
        with pytest.raises(SpotcheckError):
            _ = checked_evaluate(
                dataset=impl_dict["dataset"],
                model=impl_dict["model"],
                augmentation=impl_dict["augmentation"],
                metric=impl_dict["metric"],
            )
    else:
        with pytest.raises(SpotcheckError):
            _ = checked_evaluate(
                dataloader=impl_dict["dataloader"],
                model=impl_dict["model"],
                augmentation=impl_dict["augmentation"],
                metric=impl_dict["metric"],
            )
