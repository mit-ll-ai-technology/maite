# Show we can create a pytorch model and modify its weights within
# an Augmentation implementer. Consider image classification example

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from maite.protocols import ArrayLike
from maite.protocols.image_classification import Augmentation

# make dummy model that takes Nx5 inputs and produces a onehot
# vector of pseudoprobabilities
BATCH_SIZE = 5
dummy_model = nn.Sequential(nn.Linear(BATCH_SIZE, 5), nn.ReLU(), nn.Softmax())

# create single batch of inputs, ground-truth outputs, and metadata
input_batch = torch.rand([BATCH_SIZE, 5])
output_batch_gt = torch.tensor([10] * BATCH_SIZE)
metadata_batch = [dict() for _ in range(BATCH_SIZE)]

preds = dummy_model(input_batch)

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


# Augmentation
class WhiteboxAugmentation:
    def __init__(self, model: torch.nn.Module, attack_eps: float = 1e-3):
        # store torch model as an attribute specific to this augmentation
        self._torchmodel = model
        self.attack_eps = attack_eps

    def set_model_gradient(self, gradient: torch.Tensor):
        # set model gradient
        self.model_gradient_wrt_inputs = gradient

    def __call__(
        self, datum: tuple[ArrayLike, ArrayLike, Sequence[dict[str, Any]]]
    ) -> tuple[torch.Tensor, ArrayLike, Sequence[dict[str, Any]]]:
        # unpack tuple input
        input_batch, target_batch, metadata_batch = datum

        # ensure inputs are of type tensor
        input_batch_tn = torch.as_tensor(input_batch)
        input_batch_tn.requires_grad = True

        # (one could convert output types to a tensor, but
        # passing through as as an ArrayLike is ok too)

        # convert outputs to type tensor to fulfill promised return type
        # output_batch_tn = torch.as_tensor(output_batch_gt)
        # input_batch_tn.requires_grad = False

        # calculate loss and use gradient information to inform augmentation
        output_preds_batch = self._torchmodel(input_batch_tn)
        loss = torch.nn.CrossEntropyLoss()
        loss(output_preds_batch, target_batch)

        assert (
            input_batch_tn.grad is not None
        ), "gradient wrt model inputs must be calculated"

        input_batch_aug = (
            input_batch + input_batch_tn.grad * self.attack_eps * self.attack_eps
        )

        return (input_batch_aug, target_batch, metadata_batch)


wb: Augmentation = WhiteboxAugmentation(model=dummy_model, attack_eps=0.1)

# apply Whitebox augmentation to outputs of a dataloader
