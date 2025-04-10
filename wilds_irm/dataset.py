import enum


class Dataset(enum.StrEnum):
    """Dataset options supported for training and evaluation."""

    Waterbirds = "waterbirds"
    CelebA = "celebA"

    @property
    def group_cols(self):
        """The features of the dataset which we use to identify subgroups in the dataset."""
        match self:
            case self.Waterbirds:
                return ["background", "y"]
            case self.CelebA:
                return ["male", "y"]
