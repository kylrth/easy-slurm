from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ObjectiveConfig(ABC):
    """A base class for the objective functions we train with."""

    @abstractmethod
    def penalty(
        self,
        batch_loss: torch.Tensor,
        epoch: int,
        batch_size: int,
        lossf: nn.Module,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the penalty on the logits, given targets y."""

    @property
    @abstractmethod
    def id(self) -> str:
        """A string representing the objective and its config values."""


@dataclass
class ERMConfig(ObjectiveConfig):
    "Train using empirical risk minimization."

    def penalty(
        self,
        batch_loss: torch.Tensor,
        epoch: int,  # noqa: ARG002
        batch_size: int,
        lossf: nn.Module,  # noqa: ARG002
        logits: torch.Tensor,  # noqa: ARG002
        y: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return torch.sum(batch_loss) / batch_size

    @property
    def id(self) -> str:
        return "ERM"


@dataclass
class IRMConfig(ObjectiveConfig):
    """Train using invariant risk minimization."""

    weight: float = 100000  # multiplier for Lagrange approximation of IRM
    anneal: int = 10  # set the multiplier weight to 1 until this many epochs have passed

    def penalty(
        self,
        batch_loss: torch.Tensor,  # noqa: ARG002
        epoch: int,
        batch_size: int,
        lossf: nn.Module,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        scale = logits.new_tensor(1.0, requires_grad=True)  # place on same device as logits
        penalty = lossf(logits * scale, y.unsqueeze(-1))
        grad = torch.autograd.grad(penalty.mean(), [scale], create_graph=True)[0]
        p = torch.sum(grad**2)

        penalty = torch.sum(penalty) / batch_size

        # see https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py#L145
        p_weight = self.weight if epoch >= self.anneal else 1.0
        penalty += p_weight * p
        if p_weight > 1.0:
            # keep gradients in a reasonable range
            penalty /= p_weight
        return penalty

    @property
    def id(self) -> str:
        return f"IRMw{self.weight:06d}a{self.anneal:03d}"
