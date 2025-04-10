from pathlib import Path
from typing import TYPE_CHECKING, cast

from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset

from common import Recorder


if TYPE_CHECKING:
    import torch


class WGRecorder(Recorder):
    """Record intermediate results during the training process, including group-based statistics.

    Creates train.csv and valid.csv by default, but more output files can be created by passing
    extra headers as kwargs.

    Adds record_train and record_valid convenience methods.
    """

    # the group names
    groups: list[str]

    def __init__(
        self,
        root: Path,
        grouper: CombinatorialGrouper,
        valset: WILDSSubset,
        **other_headers: list[str],
    ):
        # get the group names in a list
        self.groups = []
        for idx in range(cast("int", grouper.cardinality.prod().item())):
            self.groups.append(grouper.group_str(idx))

        # get the quantity of each group in the validation set
        self._valcounts = self.count_groups(grouper, valset)

        self.r = Recorder(
            root,
            train=["iter", "loss", "penalty"] + ["loss_" + g for g in self.groups],
            valid=["epoch", "loss", "acc"]
            + ["loss_" + g for g in self.groups]
            + ["acc_" + g for g in self.groups],
            **other_headers,
        )

    @staticmethod
    def count_groups(grouper: CombinatorialGrouper, subset: WILDSSubset) -> list[int]:
        """Get frequency counts for the groups in the subset.

        Assumes all groups are represented.
        """
        z = cast("torch.Tensor", grouper.metadata_to_group(subset.metadata_array))

        return z.unique(sorted=True, return_counts=True)[1].tolist()

    def __enter__(self):
        self.r.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.r.__exit__(exc_type, exc_val, exc_tb)

    def record_train(
        self,
        it: int,
        loss: float,
        penalty: float,
        group_loss: list[float],
        group_counts: list[int],
    ):
        """Record the loss, penalty, and group-wise loss for this training iteration.

        `group_loss` is the sum of the loss only on examples from each group.
        `group_counts` is the number of examples in each group.
        """
        # collect group loss *per example*
        addl_args = {}
        for i, g in enumerate(self.groups):
            if group_counts[i] == 0:
                addl_args["loss_" + g] = -1.0
            else:
                addl_args["loss_" + g] = group_loss[i] / group_counts[i]

        self.r.record("train", iter=it, loss=loss, penalty=penalty, **addl_args)

    def record_valid(self, epoch: int, group_loss: list[float], group_acc_sum: list[float]):
        """Record the validation loss and accuracy (overall and group-wise) at the end of the epoch.

        `group_loss` is the sum of the loss only on examples from each group.
        `group_acc_sum` is the sum of accuracies on the examples from each group.
        """
        # collect group loss *per example*, and divide accuracy sums by the count to get percentages
        addl_args = {}
        for i, g in enumerate(self.groups):
            addl_args["loss_" + g] = group_loss[i] / self._valcounts[i]
            addl_args["acc_" + g] = group_acc_sum[i] / self._valcounts[i]

        val_total = sum(self._valcounts)
        agg_loss = sum(group_loss) / val_total
        agg_acc = sum(group_acc_sum) / val_total

        self.r.record("valid", epoch=epoch, loss=agg_loss, acc=agg_acc, **addl_args)
