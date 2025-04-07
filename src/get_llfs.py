"""
git clone https://github.com/metatensor/metatrain.git
pip install -e metatrain/
!python -m pip install pet-neighbors-convert --no-build-isolation
"""

import os
import torch


from src.dataset_operations.dataset_operations import merge_subsets, get_llf
from src.utils.consts import SPLITS


def run_get_llfs(model, dataset_path, output_folder, type):

    os.makedirs(output_folder, exist_ok=True)

    for split in SPLITS:
        print(f"Processing {split}...")

        subsets = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        subsets.sort()

        for subset in subsets:
            subset_input_path = os.path.join(dataset_path, subset, type, f"{split}.xyz")

            llfs_tensor = get_llf(model, subset_input_path)

            subset_output_path = os.path.join(output_folder, subset)
            os.makedirs(subset_output_path, exist_ok=True)

            save_path = os.path.join(subset_output_path, f"{split}.pt")
            torch.save(llfs_tensor, save_path)

            print(f"Saved to {save_path}")
