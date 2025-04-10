# ruff: noqa: T201  # this is a script, needs to print
import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import wilds
from simple_parsing import ArgumentParser, subgroups
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper, Grouper

from common import Checkpointer

from . import Dataset, Model
from .objective import ERMConfig, IRMConfig, ObjectiveConfig
from .recorder import WGRecorder


if TYPE_CHECKING:
    from wilds.datasets.wilds_dataset import WILDSDataset


# In this script we define a TrainConfig class that completely defines an experiment, and then we
# create an array of experiments each of which has an associated TrainConfig object. Then run.sh
# just has to pass an array ID to choose which experiment to run. The Dirs class is used to
# configure the folders where inputs and outputs go, and these values are read from this script's
# command-line arguments using the simple-parsing library.


@dataclass
class Dirs:
    """Directory configuration for running the script."""

    wilds: Path = Path("./data/")  # path where WILDS data is stored
    model_cache: Path | str = ""  # override the PyTorch model cache
    checkpoints: Path = Path("./checkpoints/")  # path where training checkpoints are stored
    runs: Path = Path("./runs/")  # path where statistics are written


@dataclass
class TrainConfig:  # configuration
    objective: ObjectiveConfig = subgroups(
        {"erm": ERMConfig, "irm": IRMConfig},
        ERMConfig(),
    )
    dataset: Dataset = Dataset.Waterbirds
    model: Model = Model.ResNet18
    pretrained: bool = True  # use ImageNet-1K-V1 pretrained weights from PyTorch hub
    batch_size: int = 128
    epochs: int = 30
    lr: float = 0.0005  # learning rate
    momentum: float = 0.9  # momentum for SGD
    lr_step: int = 30  # learning rate step size
    weight_decay: float = 1e-4
    seed: int = 89

    @property
    def id(self) -> str:
        """An experiment ID composed of the fields set in this config."""
        fields = [
            self.objective.id,
            self.dataset,
            self.model,
            "pret" if self.pretrained else "rand",
            f"batch{self.batch_size:03d}",
            f"epoch{self.epochs:03d}",
            f"lr{self.lr}",
            f"m{self.momentum}",
            f"lrstep{self.lr_step:03d}",
            f"wdec{self.weight_decay}",
            f"seed{self.seed:03d}",
        ]

        return "_".join(fields)

    def setup_dataset(self, ds_root: Path):
        dataset = cast("WILDSDataset", wilds.get_dataset(self.dataset, root_dir=ds_root))

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        train = dataset.get_subset("train", transform=transform)
        val = dataset.get_subset("val", transform=transform)

        grouper = CombinatorialGrouper(dataset, self.dataset.group_cols)

        return train, val, grouper

    def train(self, dirs: Dirs):
        device = "cuda:0"  # We're only doing single-GPU training

        # always set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002  # setting package seed to affect all libraries
        torch.manual_seed(self.seed)

        # set up dataset
        train, val, grouper = self.setup_dataset(dirs.wilds)
        trainloader = cast(
            "DataLoader", get_train_loader("standard", train, batch_size=self.batch_size)
        )
        valloader = cast("DataLoader", get_eval_loader("standard", val, batch_size=self.batch_size))

        # set up model
        model = self.model.init(self.pretrained, device, dirs.model_cache)

        # set up optimization
        bce_loss = nn.BCELoss(reduction="none").float()
        optim = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        sched = StepLR(optim, step_size=self.lr_step)

        # get the weights from the last existing checkpoint for this training config
        recorder = WGRecorder(dirs.runs / self.id, grouper, val)
        ckpt = Checkpointer(dirs.checkpoints / self.id)
        if ckpt.latest == 0:
            print("starting with fresh model", file=sys.stderr)
        else:
            ckpt.load_latest(model, optim, sched)
            print(f"starting from checkpoint {ckpt.latest}", file=sys.stderr)

        # train
        with (
            recorder,  # open the recorder files
            tqdm(  # start a progress bar
                desc="training",
                initial=ckpt.latest * len(trainloader),
                total=self.epochs * len(trainloader),
                mininterval=10,  # only update progress bar occasionally to reduce SLURM output
            ) as pbar,
        ):
            model.train()
            while ckpt.latest < self.epochs:
                epoch = ckpt.latest + 1
                pbar.set_description(f"train {epoch}/{self.epochs}")
                for i, (x, y, metadata) in enumerate(trainloader):
                    it = ckpt.latest * len(trainloader) + i + 1
                    x = x.to(device)
                    y = y.float().to(device)
                    z = cast("torch.Tensor", grouper.metadata_to_group(metadata)).to(device)

                    self.train_iteration(model, optim, x, y, z, epoch, it, bce_loss, recorder)
                    pbar.update()

                sched.step()

                pbar.set_description(f"valid {epoch}/{self.epochs}")
                self.validation(model, valloader, grouper, epoch, bce_loss, recorder, device)
                ckpt.checkpoint(model, optim, sched)

    def train_iteration(
        self,
        model: nn.Module,
        optim: Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        epoch: int,
        it: int,
        loss: nn.Module,
        recorder: WGRecorder,
    ):
        optim.zero_grad()

        # evaluate the model
        logits = torch.sigmoid(model(x))

        # accumulate group losses/counts (this is just straight-up loss, not necessarily the
        # training objective)
        batch_loss = loss(logits, y.unsqueeze(-1))
        agg_loss = torch.sum(batch_loss).item()

        losses = []
        counts = []
        for i in range(len(recorder.groups)):
            mask = z.eq(i).unsqueeze(-1)
            losses.append(torch.sum(batch_loss * mask).item())
            counts.append(torch.sum(mask).item())

        # This is the training objective.
        p = self.objective.penalty(batch_loss, epoch, self.batch_size, loss, logits, y)
        p.backward()
        penalty = p.item()

        recorder.record_train(it, agg_loss, penalty, losses, counts)

        optim.step()

    def validation(
        self,
        model: nn.Module,
        valloader: DataLoader,
        grouper: Grouper,
        epoch: int,
        loss: nn.Module,
        recorder: WGRecorder,
        device: str,
    ):
        with torch.no_grad():
            model.eval()

            losses = [0.0] * len(recorder.groups)
            accs = [0.0] * len(recorder.groups)

            for x, y, metadata in valloader:
                x = x.to(device)
                y = y.to(device)
                z = grouper.metadata_to_group(metadata).to(device)

                logits = torch.sigmoid(model(x))
                batch_loss = loss(logits, y.unsqueeze(-1).float())
                batch_preds = logits >= 0.5
                acc = batch_preds.squeeze(-1) == y

                # collect per-group losses and accuracies
                for i in range(len(recorder.groups)):
                    mask = z.eq(i)
                    losses[i] += torch.sum(batch_loss.squeeze(-1) * mask).detach().item()
                    accs[i] += (
                        torch.logical_and(acc, mask).sum().detach().item()
                    )  # aggregate without division

            recorder.record_valid(epoch, losses, accs)

            model.train()


# This is where you can define an array of experiments to run. Here we are sweeping over all
# combinations of various config options, but you can define whatever experiments you like here.
params = list(
    itertools.product(
        list(Dataset),  # iterate on dataset first so the lower array IDs are all one dataset
        [ERMConfig(), IRMConfig()],
        [Model.ResNet18, Model.ViT_b_32],
    )
)
exps = []
for ds, obj, m in params:
    exp = TrainConfig(
        objective=obj,
        dataset=ds,
        model=m,
        epochs=2 if ds == "celebA" else 5,
    )
    exps.append(exp)

parser = ArgumentParser(description="train models on WILDS datasets using ERM or IRM")
parser.add_argument(
    "exp",
    metavar="EXP",
    type=int,
    default=-1,
    nargs="?",
    help="experiment number to run. If -1, instead prints the experiment combinations.",
)
parser.add_arguments(Dirs, dest="directories")

args = parser.parse_args()

# If the experiment ID is not provided, we print out a list of the experiments for reference.
if args.exp == -1:
    for i, e in enumerate(exps):
        print(f"{i}: {e.id}", file=sys.stdout)
    sys.exit(2)

conf: TrainConfig = exps[args.exp]
print(f"running experiment: {conf.id}", file=sys.stderr)

# run the experiment!
dirs: Dirs = args.directories
conf.train(dirs)
