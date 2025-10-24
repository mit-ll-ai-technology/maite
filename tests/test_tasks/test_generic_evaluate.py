# --- Define general-purpose type hierarchy for use when writing static type
# checking for metacode-generated source code ---
#
# Each enum allows us to specify some class to use as a placeholder in a generated source code.
# The enum also provides simple class defs that preserve subclassing relationships.
# (Using enum permits us to take narrow arguments for generation functions instead of the much more
# open set of strings and to easily iterate using syntax like [t for t in list(InputTypes)[1:3]],
# for example)
#
# Note: In theory, we could define only one enum for use among input/target/metadata types,
# but this would make reading error messages much more confusing.
from enum import Enum
from pathlib import Path

import pytest

from maite._internals.testing.pyright import (
    PyrightOutput,
    chdir,
    list_error_messages,
    pyright_analyze,
)


class TypeEnum(Enum):
    CLS_SUB = -1
    CLS = 0
    CLS_SUP = 1

    @property
    def type_name(self) -> str:
        return {self.CLS_SUB: "ClsSub", self.CLS: "Cls", self.CLS_SUP: "ClsSup"}[self]

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.type_name}"

    @staticmethod
    def class_def_code_block():
        return """
class ClsSup: ...
class Cls(ClsSup): ...
class ClsSub(Cls): ...
        """


def gen_evaluate_static_validation_code(
    dataloader_input_typespec: TypeEnum,
    dataloader_target_typespec: TypeEnum,
    dataloader_metadata_typespec: TypeEnum,
    dataset_input_typespec: TypeEnum,
    dataset_target_typespec: TypeEnum,
    dataset_metadata_typespec: TypeEnum,
    augmentation_inputco_typespec: TypeEnum,
    augmentation_targetco_typespec: TypeEnum,
    augmentation_metadataco_typespec: TypeEnum,
    augmentation_inputcn_typespec: TypeEnum,
    augmentation_targetcn_typespec: TypeEnum,
    augmentation_metadatacn_typespec: TypeEnum,
    model_input_typespec: TypeEnum,
    model_target_typespec: TypeEnum,
    metric_target_typespec: TypeEnum,
    metric_metadata_typespec: TypeEnum,
) -> str:
    # Generate the final code using a formatted multi-line string
    return f"""
from typing import Sequence

from maite.protocols.generic import (
    Model,
    Metric,
    DataLoader,
    Dataset,
    Augmentation,
)

from maite.tasks import evaluate

from typing import cast

# --- Custom type definitions ---
{TypeEnum.class_def_code_block()}

# --- define input/target/metadata types with class relationships----
# define enough of each such that all components could be (in theory)
# concretized on different parameters on a single evaluate call

class ADataLoader(
    DataLoader[{dataloader_input_typespec.type_name},
               {dataloader_target_typespec.type_name},
               {dataloader_metadata_typespec.type_name}]
    ): ...

class ADataset(Dataset[{dataset_input_typespec.type_name},
                       {dataset_target_typespec.type_name},
                       {dataset_metadata_typespec.type_name}]): ...

class AnAugmentation(
    Augmentation[
        {augmentation_inputco_typespec.type_name},
        {augmentation_targetco_typespec.type_name},
        {augmentation_metadataco_typespec.type_name},
        {augmentation_inputcn_typespec.type_name},
        {augmentation_targetcn_typespec.type_name},
        {augmentation_metadatacn_typespec.type_name},
    ]
): ...

class AModel(Model[{model_input_typespec.type_name},
                   {model_target_typespec.type_name}]): ...

class AMetric(Metric[{metric_target_typespec.type_name}, {metric_metadata_typespec.type_name}]): ...

# Cast simple integers as above types (we only want to check static validity)
model: AModel = cast(AModel, 0)
metric: AMetric = cast(AMetric, 1)
dataLoader: ADataLoader = cast(ADataLoader, 2)
augmentation: AnAugmentation = cast(AnAugmentation, 3)
dataset: ADataset = cast(ADataset, 4)

evaluate(
    model=model,
    metric=metric,
    dataset=dataset,
    dataloader=dataLoader,
    augmentation=augmentation,
)
""".strip()  # Remove leading/trailing whitespace


# The following test statically validates a baseline evaluate call along with
# univariate perturbations to all inputs in subclass and superclass "directions"

# An example univariate perturbation would be passing evaluate a 'Model' whose
# input type is a subclass of the input type used by all other components. We
# ensure evaluate calls are statically valid/invalid when we would expect them
# to be by repeating this perturbation process for all generic typevars used with
# all components and for both subclass and superclass perturbation "directions".

TYPESPECS_TO_TEST = (
    [[TypeEnum(0) for _ in range(16)]]
    + [[TypeEnum(0) if i != j else TypeEnum(-1) for i in range(16)] for j in range(16)]
    + [[TypeEnum(0) if i != j else TypeEnum(1) for i in range(16)] for j in range(16)]
)
xpass_indices = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 26, 27, 28, 29, 31])
xfail_indices = set(range(32)) - xpass_indices

TYPESPECS_TO_TEST_PASS = [TYPESPECS_TO_TEST[pi] for pi in xpass_indices]
TYPESPECS_TO_TEST_FAIL = [pytest.param(*TYPESPECS_TO_TEST[fi]) for fi in xfail_indices]


@pytest.mark.slow
@pytest.mark.parametrize(
    """
    dataloader_input_typespec,
    dataloader_target_typespec,
    dataloader_metadata_typespec,
    dataset_input_typespec,
    dataset_target_typespec,
    dataset_metadata_typespec,
    augmentation_inputco_typespec,
    augmentation_targetco_typespec,
    augmentation_metadataco_typespec,
    augmentation_inputcn_typespec,
    augmentation_targetcn_typespec,
    augmentation_metadatacn_typespec,
    model_input_typespec,
    model_target_typespec,
    metric_target_typespec,
    metric_metadata_typespec,
    """,
    TYPESPECS_TO_TEST_PASS,
)
def test_static_evaluate(
    dataloader_input_typespec: TypeEnum,
    dataloader_target_typespec: TypeEnum,
    dataloader_metadata_typespec: TypeEnum,
    dataset_input_typespec: TypeEnum,
    dataset_target_typespec: TypeEnum,
    dataset_metadata_typespec: TypeEnum,
    augmentation_inputco_typespec: TypeEnum,
    augmentation_targetco_typespec: TypeEnum,
    augmentation_metadataco_typespec: TypeEnum,
    augmentation_inputcn_typespec: TypeEnum,
    augmentation_targetcn_typespec: TypeEnum,
    augmentation_metadatacn_typespec: TypeEnum,
    model_input_typespec: TypeEnum,
    model_target_typespec: TypeEnum,
    metric_target_typespec: TypeEnum,
    metric_metadata_typespec: TypeEnum,
):
    static_evaluate(
        dataloader_input_typespec,
        dataloader_target_typespec,
        dataloader_metadata_typespec,
        dataset_input_typespec,
        dataset_target_typespec,
        dataset_metadata_typespec,
        augmentation_inputco_typespec,
        augmentation_targetco_typespec,
        augmentation_metadataco_typespec,
        augmentation_inputcn_typespec,
        augmentation_targetcn_typespec,
        augmentation_metadatacn_typespec,
        model_input_typespec,
        model_target_typespec,
        metric_target_typespec,
        metric_metadata_typespec,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    """
    dataloader_input_typespec,
    dataloader_target_typespec,
    dataloader_metadata_typespec,
    dataset_input_typespec,
    dataset_target_typespec,
    dataset_metadata_typespec,
    augmentation_inputco_typespec,
    augmentation_targetco_typespec,
    augmentation_metadataco_typespec,
    augmentation_inputcn_typespec,
    augmentation_targetcn_typespec,
    augmentation_metadatacn_typespec,
    model_input_typespec,
    model_target_typespec,
    metric_target_typespec,
    metric_metadata_typespec,
    """,
    TYPESPECS_TO_TEST_FAIL,
)
def test_static_evaluate_raises(
    dataloader_input_typespec: TypeEnum,
    dataloader_target_typespec: TypeEnum,
    dataloader_metadata_typespec: TypeEnum,
    dataset_input_typespec: TypeEnum,
    dataset_target_typespec: TypeEnum,
    dataset_metadata_typespec: TypeEnum,
    augmentation_inputco_typespec: TypeEnum,
    augmentation_targetco_typespec: TypeEnum,
    augmentation_metadataco_typespec: TypeEnum,
    augmentation_inputcn_typespec: TypeEnum,
    augmentation_targetcn_typespec: TypeEnum,
    augmentation_metadatacn_typespec: TypeEnum,
    model_input_typespec: TypeEnum,
    model_target_typespec: TypeEnum,
    metric_target_typespec: TypeEnum,
    metric_metadata_typespec: TypeEnum,
):
    with pytest.raises(ValueError):
        static_evaluate(
            dataloader_input_typespec,
            dataloader_target_typespec,
            dataloader_metadata_typespec,
            dataset_input_typespec,
            dataset_target_typespec,
            dataset_metadata_typespec,
            augmentation_inputco_typespec,
            augmentation_targetco_typespec,
            augmentation_metadataco_typespec,
            augmentation_inputcn_typespec,
            augmentation_targetcn_typespec,
            augmentation_metadatacn_typespec,
            model_input_typespec,
            model_target_typespec,
            metric_target_typespec,
            metric_metadata_typespec,
        )


def static_evaluate(
    dataloader_input_typespec: TypeEnum,
    dataloader_target_typespec: TypeEnum,
    dataloader_metadata_typespec: TypeEnum,
    dataset_input_typespec: TypeEnum,
    dataset_target_typespec: TypeEnum,
    dataset_metadata_typespec: TypeEnum,
    augmentation_inputco_typespec: TypeEnum,
    augmentation_targetco_typespec: TypeEnum,
    augmentation_metadataco_typespec: TypeEnum,
    augmentation_inputcn_typespec: TypeEnum,
    augmentation_targetcn_typespec: TypeEnum,
    augmentation_metadatacn_typespec: TypeEnum,
    model_input_typespec: TypeEnum,
    model_target_typespec: TypeEnum,
    metric_target_typespec: TypeEnum,
    metric_metadata_typespec: TypeEnum,
):
    # generate code
    val_code_str: str = gen_evaluate_static_validation_code(
        dataloader_input_typespec,
        dataloader_target_typespec,
        dataloader_metadata_typespec,
        dataset_input_typespec,
        dataset_target_typespec,
        dataset_metadata_typespec,
        augmentation_inputco_typespec,
        augmentation_targetco_typespec,
        augmentation_metadataco_typespec,
        augmentation_inputcn_typespec,
        augmentation_targetcn_typespec,
        augmentation_metadatacn_typespec,
        model_input_typespec,
        model_target_typespec,
        metric_target_typespec,
        metric_metadata_typespec=metric_metadata_typespec,
    )

    # write code to temp file
    with chdir():
        cwd = Path.cwd()
        out_path = cwd / "test_gen_evaluate.py"
        out_path.write_text(val_code_str, encoding="utf-8")

        # run pyright_analyze
        scan: PyrightOutput = pyright_analyze(out_path)[0]
        if scan["summary"]["errorCount"] != 0:
            raise ValueError(
                "\n"
                + "Pyright error in generic evaluate test case"
                + "\n"
                + "\n".join(list_error_messages(scan))
            )
            # Maybe publish specific failure case for traceability?
            # Pytest should share failing args though, and 15-element
            # list of enums wouldn't be terribly informative anyway
