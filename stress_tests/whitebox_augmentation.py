# %% [markdown]
# ## Whitebox augmentation demo
#
# We use a MAITE image-classification `Augmentation` to represent a simple
# adversarial attack on model input data. Because the attack depends on
# gradient information that is not guaranteed to be available for MAITE
# `Model` objects, we must get the information in an application-specific
# way. In this case, we write the `Augmentation` implementer class such that
# it has access to the underlying (framework-specific) model. This way, the
# `Augmentation` implementer can access model gradients internally within its
# `__call__` method and after construction the implementer can be treated as
# any other implementer of `Augmentation`.
#
# In this example, we consider the image classification domain
# where input and targets from the a prediction model are both tensors.
# %% [markdown]
# ## Setup
# %%
from __future__ import (  # permit use of tuple/dict as generic typehints in 3.8
    annotations,
)

import copy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, Tuple

import torch
from torch import nn

from maite.protocols import ArrayLike
from maite.protocols.image_classification import Augmentation


# %% [markdown]
# ## Define a simple protocol for a broad set of attacks
# This isn't strictly necessary, but helpful for extensibility.
#
# Any interface expecting an instance of this protocol class will be
# able to handle any implementer (structural subtype), so implementations
# of attacks can be modified/rewritten without modifying classes
# that are expected to use those objects.
# %%
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


# %% [markdown]
# ## Define a simple implementer of above protocol class
# %%
@dataclass
class DumbAttack:
    """
    Very basic implementer of above ImageClassifierAttack protocol
    """

    name: str

    def __call__(
        self,
        model: torch.nn.Module,
        input_batch: ArrayLike,
        target_batch: ArrayLike,
    ) -> torch.Tensor:
        """
        Given a torch model, a model input batch, and a model target batch
        (i.e. ground truth) calculate an adversarial perturbation that can be
        added to the input tensor to form an adversarial input.
        """

        # type-narrow inputs to type tensor
        input_batch_tn = torch.as_tensor(input_batch)
        input_batch_tn.requires_grad = True

        # type-narrow targets to type tensor
        target_batch_tn = torch.as_tensor(target_batch)

        preds = model(input_batch_tn)

        # calculate some simple loss
        loss = torch.sum(
            torch.nn.functional.binary_cross_entropy(preds, target_batch_tn)
        )
        loss.backward()

        assert input_batch_tn.grad is not None

        return input_batch_tn.grad * 1e-4


# TODO: Try to demonstrate a standard approach to implement protocols that permit
#       structural subclass checks at that class object level (i.e. using
#       "issubclass(SomeUserClass, SomeProtocol)" and dont require instantiation.
#       (i.e. isinstance(SomeUserClass(...), SomeProtocol). Otherwise, introspection
#       and inference tools wont be able to verify protocol compatibility without
#       instantiating. This inferrence ability is a huge potential gain.


# %% [markdown]
# ## Define a "whitebox augmentation" class
# The class will store framework-specific model while implementing MAITE
# `Augmentation` protocol


# %%
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

        # type-narrow targets to type tensor
        target_batch_tn = torch.as_tensor(target_batch)

        attack_perturbation = self.attack(self.model, input_batch_tn, target_batch_tn)
        input_batch_aug = input_batch_tn + attack_perturbation

        # Modify returned metadata object to record any important
        # aspects of this augmentation
        metadata_batch_aug = copy.deepcopy(metadata_batch)
        for i, datum_metadata in enumerate(metadata_batch_aug):
            if "aug_applied" not in datum_metadata.keys():
                datum_metadata["augs_applied"] = list()

                datum_metadata["augs_applied"].append(
                    {
                        "name": self.attack.name,
                        "mean_perturbation": torch.mean(attack_perturbation[i]).numpy(),
                    }
                )

        return (input_batch_aug, target_batch_tn, metadata_batch_aug)


# %% [markdown]
# ## Test the augmentation
# Create dummy torch module and batch of input/target/metadata

# %%
# make dummy model that takes Nx5 inputs and produces a onehot
# vector of pseudoprobabilities
BATCH_SIZE = 4
H_IMG = 32
W_IMG = 32
C_IMG = 3
N_CLASSES = 5

dummy_model = nn.Sequential(
    nn.Flatten(), nn.Linear(H_IMG * W_IMG * C_IMG, N_CLASSES), nn.ReLU(), nn.Softmax()
)

# %%
# Apply a WhiteboxAugmentation to a batch


# create instance of WhiteboxAugmentation class
wb_aug: Augmentation = WhiteboxAugmentation(
    model=dummy_model, attack=DumbAttack(name="silly_attack")
)

# create a 'dummy' datum batch
datum_batch: Tuple[torch.Tensor, torch.Tensor, Sequence[dict[str, Any]]] = (
    torch.rand((BATCH_SIZE, C_IMG, H_IMG, W_IMG)),
    torch.eye(BATCH_SIZE, N_CLASSES),
    [dict() for _ in range(BATCH_SIZE)],
)

# apply augmentation
datum_batch_aug = wb_aug(datum_batch)

# %% [markdown]
# ## Print result of augmentation

# %%
# unpack datums
# TODO: consider whether tuple of iterables or iterable of tuples is more convenient
#       as a batch format. Tuple of iterables seems to require below unpacking

model_input_batch_aug, model_target_batch_aug, md_batch_aug = datum_batch_aug
model_input_batch, model_target_batch, md_batch = datum_batch

print("Results of augmentation (by datum)")
for model_input_aug, model_target_aug, md_aug, model_input, model_target, md in zip(
    model_input_batch_aug,
    model_target_batch_aug,
    md_batch_aug,
    model_input_batch,
    model_target_batch,
    md_batch,
):
    print(f"model input:\n {model_input}")
    print(f"model input (augmented):\n {model_input_aug}")
    print(f"datum metadata:\n {md_aug}")
    print("\n")

# %% [markdown]
#
# ## Some Design Considerations:
#
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
