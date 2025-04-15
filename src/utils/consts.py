import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["train", "val", "test"]
BATCH_SIZE = 64

# mad dataset
MAD_FOLDER = "data/v1.5"
TYPE = "cleaned-pbc"
PROJECTION_FOLDER = "data/atom_smap"
MAD_LFF_OUTPUT_FOLDER = f"{MAD_FOLDER}_{TYPE}_llfs"

# mptraj
MPTRAJ_FOLDER = "data/mptraj"
MPTRAJ_LFF_OUTPUT_FOLDER = f"{MPTRAJ_FOLDER}_llfs"


def get_subsets(path="data/v1.5_cleaned-pbc_llfs"):
    subsets = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    subsets.sort()
    return subsets


SUBSETS = get_subsets()

PARAMS = {"lr": 0.001, "weight_decay": 1e-4, "epochs": 1000, "batch_size": 64}
