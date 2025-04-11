import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["train", "val", "test"]
BATCH_SIZE = 64

DATASET_FOLDER = "data/v1.5"
TYPE = "cleaned-pbc"
OUTPUT_FOLDER = f"{DATASET_FOLDER}_{TYPE}_merged"

PROJECTION_FOLDER = "data/atom_smap"

LFF_OUTPUT_FOLDER = f"{DATASET_FOLDER}_{TYPE}_llfs"


def get_subsets(path="data/v1.5_cleaned-pbc_llfs"):
    subsets = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    subsets.sort()
    return subsets


SUBSETS = get_subsets()
