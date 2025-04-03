"""
git clone https://github.com/metatensor/metatrain.git
pip install -e metatrain/
!python -m pip install pet-neighbors-convert --no-build-isolation
"""

import os
import torch

from metatrain.experimental.nativepet import NativePET

from src.dataset_operations.dataset_operations import merge_subsets, get_llf
from src.utils.consts import (
    DATASET_FOLDER,
    OUTPUT_FOLDER,
    DEVICE,
    TYPE,
    LFF_OUTPUT_FOLDER,
    SPLITS,
)


if __name__ == "__main__":
    model = NativePET.load_checkpoint("pet-mad-latest.ckpt").eval().to(DEVICE)

    merge_subsets(DATASET_FOLDER, OUTPUT_FOLDER, TYPE)

    os.makedirs(LFF_OUTPUT_FOLDER, exist_ok=True)

    for split in SPLITS:
        print(f"Processing {split}...")

        dataset_path = os.path.join("v1.5_cleaned-pbc_merged", f"{split}.xyz")

        llfs_tensor = get_llf(model, dataset_path)
        torch.save(llfs_tensor, os.path.join(LFF_OUTPUT_FOLDER, f"{split}.pt"))
