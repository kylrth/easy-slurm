# easy SLURM

This repo contains high-quality example code that demonstrates (and documents) good practices for machine learning experiments on a SLURM cluster. These examples were developed on [DRAC](https://alliancecan.ca/en) infrastructure and make a few assumptions about the cluster's configuration:

- [**filesystems**](https://docs.alliancecan.ca/wiki/Storage_and_file_management): `/home`, `/project`, and `/scratch` are all backed by a distributed filesystem (Lustre), meaning that reading and writing many small files incurs a lot of overhead. Scripts in this repo are thus optimized to reduce the number of read and write operations to those locations.
  - `/home` is for source code (including this repo), environments, configuration files.
  - `/project` is for shared resources that change infrequently, such as datasets that your entire group uses, or base models you'll be training from.
  - `/scratch` is for temporarily storing experiment outputs, checkpoints, anything that can be recreated. This location is not backed up and files are deleted after a few months.
  - `$SLURM_TMPDIR` exists only on compute nodes, and contains the path to a fast, local filesystem. Most disk I/O performed on compute nodes should be done here.
- **Internet access is disabled** on compute nodes in [Béluga](https://docs.alliancecan.ca/wiki/B%C3%A9luga/en) and [Narval](https://docs.alliancecan.ca/wiki/Narval/en). All of these job scripts will run correctly without internet access, because the setup scripts download all necessary data and dependencies beforehand. Even if your cluster offers internet access on compute nodes, it is good practice to not waste job time waiting for network resources.

## running the examples

Each folder in this repo contains a setup script and a run script. On DRAC clusters, the only necessary modification is updating the `$PROJECT` variable in both scripts to point to your personal project directory. The setup script should be run directly on the login node, e.g. `./wilds_irm/setup.sh`. The setup script only needs to be run once to download dependencies and set up the environment. After this, a job can be submitted using `sbatch`: `sbatch ./wilds_irm/run.sh`. **Before submitting, please review the configuration options at the top of the run script, to make sure you know what volume of jobs you're about to submit.**

## examples overview

- `wilds_irm/`: finetune a pretrained ResNet vision model on the WILDS dataset. Demonstrates:
  - essential SLURM job configuration options
  - dataset management
  - base model management
  - Python environment management with `virtualenv` and `pip`, using manually downloaded wheels for packages not provided on the cluster
  - CUDA-accelerated PyTorch
  - the [`simple-parsing`](https://github.com/lebrice/SimpleParsing) library

## TODO

- add some example analysis code for `wilds_irm/`
- use `uv`
- example of finetuning using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
