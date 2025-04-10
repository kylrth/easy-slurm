from pathlib import Path
from typing import Any, Protocol

import torch


class Stateful(Protocol):
    def load_state_dict(self, state_dict: dict[str, Any]): ...

    def state_dict(self) -> dict[str, Any]: ...


class Checkpointer:
    """Store PyTorch checkpoints in root, loading the lastest checkpoint on init."""

    latest: int  # number of completed epochs

    def __init__(self, root: Path, keep: int = 5):
        self.root = root
        self.keep = keep

        self.root.mkdir(parents=True, exist_ok=True)

        # find the latest checkpoint by number
        self.latest = 0
        for f in self.root.glob("*.pt"):
            num = int(f.name[:-3])  # remove .pt
            self.latest = max(self.latest, num)

    def load_latest(self, *modules: torch.nn.Module | Stateful):
        """Load the latest checkpoint into the provided modules."""
        ckpt = torch.load(self.root / f"{self.latest}.pt")
        for m, d in zip(modules, ckpt):
            m.load_state_dict(d)

    def checkpoint(self, *modules: torch.nn.Module | Stateful):
        """Store a checkpoint for a newly completed epoch."""
        self.latest += 1
        torch.save([m.state_dict() for m in modules], self.root / f"{self.latest}.pt")

        if self.latest > self.keep:
            # delete the checkpoint from self.keep epochs ago
            (self.root / f"{self.latest - self.keep}.pt").unlink(missing_ok=True)
