import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from io import TextIOWrapper


class Recorder:
    """Iteratively record data to (multiple) CSV files."""

    def __init__(self, root: Path, **headers: list[str]):
        """`headers` is a dictionary of all the headers for each CSV file to create.

        Each key creates a CSV file "{key}.csv".
        """
        self.root = root
        self.headers = headers
        self.fs: dict[str, TextIOWrapper] = {}
        self.csvs: dict[str, csv.DictWriter] = {}

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)

        for name in self.headers:
            # append to files in case we're starting from a checkpoint
            self.fs[name] = (self.root / f"{name}.csv").open("a+", newline="")
            self.csvs[name] = csv.DictWriter(self.fs[name], self.headers[name])
            self.csvs[name].writeheader()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for f in self.fs.values():
            f.close()

        return False

    def record(self, key: str, **values: Any):
        """Record values in the CSV corresponding to `key`."""
        self.csvs[key].writerow(values)
