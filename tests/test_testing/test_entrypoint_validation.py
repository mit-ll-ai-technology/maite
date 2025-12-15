from importlib.metadata import entry_points

from maite._internals.testing.project import (
    statically_verify_component_entrypoint_against_protocol,
    statically_verify_exposed_component_entrypoints,
)


def test_component_entrypoint_static_verification():
    # check that the object detection model advertised by maite's pyproject.toml
    # (YoloObjectDetector) is statically valid
    od_model_results: dict[str, bool] = (
        statically_verify_component_entrypoint_against_protocol(
            protocol_module="maite.protocols.object_detection",
            protocol_name="Model",
            package_name="maite",
        )
    )

    assert len(od_model_results) == 1
    assert all([v for v in od_model_results.values()])

    # check that the image classification metric advertised by maite's pyproject.toml
    # (TMClassificationMetric) is statically valid
    ic_metric_results: dict[str, bool] = (
        statically_verify_component_entrypoint_against_protocol(
            protocol_module="maite.protocols.image_classification",
            protocol_name="Metric",
            package_name="maite",
        )
    )

    assert len(ic_metric_results) == 1
    assert all([v for v in ic_metric_results.values()])

    # check that the object detection metric advertised by maite's pyproject.toml
    # (TMDetectionMetric) is statically valid
    od_metric_results: dict[str, bool] = (
        statically_verify_component_entrypoint_against_protocol(
            protocol_module="maite.protocols.object_detection",
            protocol_name="Metric",
            package_name="maite",
        )
    )

    assert len(od_metric_results) == 1
    assert all([v for v in od_metric_results.values()])


def test_task_entrypoint_advertisement():
    """Verify that tasks exposed by MAITE are visible to querying packages"""

    # check we have 4 total tasks advertised
    eps = entry_points(group="maite.tasks")
    assert len(eps) == 4

    # test their names and module locations are where we expect
    expected_task_names = [
        "augment_dataloader",
        "evaluate",
        "evaluate_from_predictions",
        "predict",
    ]

    expected_task_module_locs = ["maite.tasks"] * 4

    for ep, expected_task_name, expected_task_module_loc in zip(
        eps, expected_task_names, expected_task_module_locs
    ):
        obj_module, obj_name = ep.module, ep.attr
        assert obj_module == expected_task_module_loc
        assert obj_name == expected_task_name


def test_verify_exposed_component_entrypoints():
    """Test that the utility looking at all exposed entrypoints in a package
    is working by applying it to MAITE"""

    exposed_component_verification_results: dict[str, bool] = (
        statically_verify_exposed_component_entrypoints(package_name="maite")
    )

    assert len(exposed_component_verification_results) == 3
    assert all([v for v in exposed_component_verification_results.values()])
