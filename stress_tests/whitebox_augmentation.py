# Show we can create a pytorch model and modify its weights within
# an Augmentation implementer. Consider image classification example

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from maite.protocols import ArrayLike
from maite.protocols.image_classification import Augmentation

# Potential methods to leverage model gradient information in augmentation
# __call__ method within maite-compliant implementers.
#
# In the below, "component implementers" are developers writing augmentation
# classes that implement MAITE augmentations and "application developers" are
# developers intending to leverage augmentation implementers for specific
# applications.
#
# 1) Store framework-specific model in augmentation implementer instance,
#    component implementers leverage framework-specific model as needed
#    within Augmentation.__call__ method.
#
#   1a) Require application developer populate a framework-specific attribute
#   within implementer augmentation class instance before use. Component implementers
#   store supported framework-specific objects internally to their implementer class
#   instances and are in charge of using those models to inform Augmentation.__call__
#   method (required types of internal model-like objects could be enforced in
#   class constructor or mutator methods that take framework-specific types
#   in their argument lists.)
#
#   Pros:
#   -----
#   - Augmentation protocol itself remains model-agnostic and minimal
#     (this agrees with idea that an augmentation shouldn't generally
#     care about framework of model, but also shouldn't preclude it.)
#
#   - Requirements for a given augmentation implementer would be clear
#     to application developer at develop-time, rather than only at runtime.
#     (e.g. 'to construct augmentation X, I need a model of type Y')
#
#   Cons:
#   -----
#   - Application developers may not be able to apply/construct particular
#     augmentation implementers if they can't populate some internal model
#     object.
#
#   - Model gradient access is outsourced to component implementers since
#     there is no standard way to access MAITE-Model gradients.
#
#   1b) Socialize *convention* that implementer methods should take maite-compliant
#       components AND that implementers should be able to downselect to internally
#       required types.
#
#   Pros:
#   -----
#   - Application developers would have *the appearance* of compatibilty with any
#     maite-compliant models (this may be considered a con)
#
#   Cons:
#   -----
#   - More work for implementers to type-narrow that is unenforceable (nothing says
#     a "maite compliant" model can't have any particular field type or have a
#     method that takes any particular argument type.)
#
#   - MAITE-compliant augmentations could have more confusing runtime errors annd
#     would have implicitly unsupported model frameworks. (This is worse than
#     explicitly unsupported model frameworks; More generally, incompatibilities
#     discoverable at development-time are preferable to incompatibilities
#     discoverable only at runtime.) Since implementers would certainly only support
#     certain native frameworks, any typenarrowing from MAITE-compliant to
#     those supported frameworks would potentially result in runtime errors or
#     at least no-ops.
#
# 2) Standardize ANY required access to gradients over all maite-compliant Models
#
#   - MAITE compliant models would require implementers to translate gradient
#     information into a standard form. MAITE team would decide on how to represent
#     certain aspects of a Model (e.g. gradients, layers) into a standard form and
#     MAITE prescribe a set of new required methods of all MAITE models that return
#     these standard types.
#
#   Pros:
#   -----
#   - If implemented, would permit component implementers to write framework-
#     agnostic implementers that only interacted with components via
#     maite-guaranteed attributes. (This would be unenforceable, and thus
#     opt-in by component implementers.)
#   - For framework-agnostic component implementers, application developers
#     wouldn't need to make any framework-specific objects to construct components
#
#   Cons:
#   -----
#   - This puts more onus on component implementers to write implementations for
#     translating more detailed model information into "standard" form. Many may
#     decide not to.
#   - Some models will not have gradient information anyway, despite being MAITE-Models.
#     Component implementers would have to decide how to handle this (runtime error
#     or warning+no-op) but the constraint of gradient access for augmentation
#     implementer use is unavoidable.
#   - Gradient access within each framework is highly-specialized and variable even
#     problem-to problem. Even within Pytorch, consider that some gradient information
#     in pytorch-based models may not have been calculated (requires_grad=False), or that deciding
#     on a representation for gradient information is difficult-should all ArrayLike variables
#     have an optional 'grad' field?
#   - Framework specific optimizations would likely still require application developer
#     interact with framework-specific model/data anyway. So there isn't as much benefit.
#   - This requires MAITE architects to standardize more types and probably make heavier
#     protocols.


# --- Define a protocol that is satisfied by a broad set of attacks ---
# (Not strictly necessary, but helpful for extensibility, classes that
# take instances of this type will be able to handle any implementer.)
class ImageClassifierAttack(Protocol):
    """
    Protocol defining an interface that might be satisfied by an attack on an
    image classifier.
    """

    def __call__(
        self,
        model: torch.nn.Module,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @property
    def name(self) -> str:
        ...


# --- Define a simple implementer of the above Protocol ---
@dataclass
class DumbAttack:
    """
    Very basic implementer of above ImageClassifierAttack protocol
    """

    name: str

    def __call__(
        self,
        model: torch.nn.Module,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given a torch model, a model input batch, and a model target batch
        (i.e. ground truth) calculate an adversarial perturbation that can be
        added to the input tensor to form an adversarial input.
        """

        # interact with input/target or model to inform an attack,
        # we just do something silly for demonstration

        return torch.Tensor(input_batch * 0.1)


# TODO: Try to demonstrate a standard approach to implement protocols that permit
#       structural subclass checks at that class object level (i.e. using
#       "issubclass(SomeUserClass, SomeProtocol)" and dont require instantiation.
#       (i.e. isinstance(SomeUserClass(...), SomeProtocol). Otherwise, introspection
#       and inference tools wont be able to verify protocol compatibility without
#       instantiating. This inferrence ability is a huge potential gain.

# --- Create a "whitebox" augmentation that stores framework specific model ---
# --- and attack while still satisfying Augmentation Protocol


# Create an Augmentation that takes anything satisfying this ImageClassifierAttack
# object in its constructor and uses it within its __call__ method. After it is
# constructed, the user can treat it like any other implementer of the Augmentation
# protocol.
class WhiteboxAugmentation:
    """
    Apply an image classifier attack
    """

    def __init__(self, model: torch.nn.Module, attack: ImageClassifierAttack):
        # store torch model as an attribute specific to this augmentation
        self.attack = attack
        self.model = model

    def __call__(
        self, datum: tuple[ArrayLike, ArrayLike, Sequence[dict[str, Any]]]
    ) -> tuple[torch.Tensor, torch.Tensor, Sequence[dict[str, Any]]]:
        # unpack tuple input
        input_batch, target_batch, metadata_batch = datum

        # type-narrow inputs to type tensor
        input_batch_tn = torch.as_tensor(input_batch)

        # type-narrow outputs to type tensor
        target_batch_tn = torch.as_tensor(target_batch)

        attack_perturbation = self.attack(self.model, input_batch_tn, target_batch_tn)
        input_batch_aug = input_batch_tn + attack_perturbation

        metadata_batch_aug = copy.deepcopy(metadata_batch)
        for datum_metadata in metadata_batch_aug:
            if "aug_applied" not in datum_metadata.keys():
                datum_metadata["augs_applied"] = list()

            rand_val = random.random()  # We can log datum-specific values in metadata
            datum_metadata["augs_applied"].append(
                {"name": self.attack.name, "rand_val": rand_val}
            )

        return (input_batch_aug, target_batch_tn, metadata_batch_aug)


# --- Create dummy torch module ---

# make dummy model that takes Nx5 inputs and produces a onehot
# vector of pseudoprobabilities
BATCH_SIZE = 5
H_IMG = 6
W_IMG = 7

dummy_model = nn.Sequential(nn.Linear(BATCH_SIZE, 5), nn.ReLU(), nn.Softmax())

# create single batch of inputs, ground-truth outputs, and metadata
input_batch = torch.rand([BATCH_SIZE, 5])
output_batch_gt = torch.tensor([10] * BATCH_SIZE)
metadata_batch = [dict() for _ in range(BATCH_SIZE)]

# --- Apply Whitebox augmentation to a batch ---

# create instance of WhiteboxAugmentation class
wb_aug: Augmentation = WhiteboxAugmentation(
    model=dummy_model, attack=DumbAttack(name="silly_attack")
)

# create a 'dummy' datum batch
datum_batch: Tuple[torch.Tensor, torch.Tensor, Sequence[dict[str, Any]]] = (
    torch.tensor(
        np.arange(BATCH_SIZE * H_IMG * W_IMG).reshape(BATCH_SIZE, H_IMG, W_IMG)
    ),
    torch.tensor(
        np.arange(BATCH_SIZE * H_IMG * W_IMG).reshape(BATCH_SIZE, H_IMG, W_IMG)
    ),
    [dict() for _ in range(BATCH_SIZE)],
)

# apply augmentation
datum_batch_aug = wb_aug(datum_batch)

# --- Print result of augmentation ---

# unpack datums
# TODO: consider whether tuple of iterables or iterable of tuples is more convenient
#       as a batch format. Tuple of iterables seems to require below unpacking
model_input_batch_aug, model_output_batch_aug, md_batch_aug = datum_batch_aug
model_input_batch, model_output_batch, md_batch = datum_batch

for model_input_aug, model_output_aug, md_aug, model_input, model_output, md in zip(
    model_input_batch_aug,
    model_output_batch_aug,
    md_batch_aug,
    model_input_batch,
    model_output_batch,
    md_batch,
):
    print(f"{model_input}")
    print(f"{model_input_aug}")
    print(f"{md_aug}")
