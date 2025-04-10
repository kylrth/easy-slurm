#!/bin/bash

# This script stores environment setup commands that need to be run only once on a login node when
# setting up the project. Update this script to ensure that anyone running this project for the
# first time is able to download and set up all datasets, models, and code necessary to run the
# experiments. This saves compute node time that might be spent redoing this preparation work, and
# is essential on some clusters because the compute nodes don't have internet access.

set -e

# load SLURM modules necessary for the Python environment we are building
# See here for information about the Lmod module system: https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en
# They need to be loaded in this order because latter ones depend on former ones.
module load gcc arrow python/3.11 scipy-stack

# TODO: change the directories below to the appropriate locations on your cluster
SOURCE=~/easy-slurm/wilds_irm  # location of this example code
PROJECT=~/projects/def-sponsor00/"$USER"/easy-slurm/wilds_irm  # location for project data, pre-trained models, etc.

# install a Python environment for data analysis on the login node
# We will set up this environment with exactly the same packages as the one we use on compute nodes,
# but for performance reasons we will have compute jobs build their own Python virtual environments
# each time.
VENV="$SOURCE"/.venv
virtualenv "$VENV"
source "$VENV"/bin/activate

# On DRAC, pip gets packages from two locations: PyPI and the DRAC "wheelhouse", a large set of
# precompiled wheels cached on the cluster. It is generally better to use these wheels unless you
# need features from a package version that is only available on PyPI. To force pip to only use
# wheels from the wheelhouse, add `--no-index` to the pip command.
# Here is the DRAC documentation for installing Python packages: https://docs.alliancecan.ca/wiki/Python#Installing_packages
# Here is the list of wheels available in the DRAC wheelhouse: https://docs.alliancecan.ca/wiki/Python#Available_wheels
# In this example, we maintain two requirements files, one containing the packages and versions we
# want to use from the cluster, and the other containing the packages and versions which we get from
# PyPI.

# install packages that are cached on the cluster
pip install --no-index -r "$SOURCE"/requirements_slurm.txt

# download wheels for packages that are not cached on the cluster
mkdir -p "$PROJECT"/wheels
pushd "$PROJECT"/wheels
pip download --no-deps -r "$SOURCE"/requirements_wheel.txt  # Any transitive dependencies not on DRAC should be added explicitly in this file.
                                                            # We use --no-deps to avoid upgrading packages we already got from DRAC.
pip install --no-index *  # also install the wheels in our login node Python environment
popd

# cache pretrained models in a folder in $PROJECT
python -c '
import torch
import torchvision.models as models

torch.hub.set_dir("'"$PROJECT"'/torch_cache")

models.resnet18(weights="IMAGENET1K_V1")
models.resnet34(weights="IMAGENET1K_V1")
models.resnet50(weights="IMAGENET1K_V1")
models.vit_b_32(weights="IMAGENET1K_V1")
models.vit_l_32(weights="IMAGENET1K_V1")
'

# store the WILDS datasets in a tar file in $PROJECT to copy to compute nodes
# It is almost always better to store datasets in tar files, because copying a single tar file is
# much more efficient on a distributed filesystem than copying the thousands of small files it
# contains. Here we download the datasets to a temporary directory, write it all to a tar file in
# $PROJECT, and then delete the original download.
DATADIR=$(mktemp -d)
python -c '
from wilds import get_dataset

get_dataset("celebA", root_dir="'"$DATADIR"'", download=True)
get_dataset("waterbirds", root_dir="'"$DATADIR"'", download=True)
'
pushd "$DATADIR"
tar -cf "$PROJECT"/data.tar --use-compress-program="pigz -p 4" .
popd
rm -rf "$DATADIR"
