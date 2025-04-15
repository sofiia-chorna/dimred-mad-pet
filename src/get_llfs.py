"""
git clone https://github.com/metatensor/metatrain.git
pip install -e metatrain/
!python -m pip install pet-neighbors-convert --no-build-isolation
"""

import os

import torch

from src.dataset_operations.dataset_operations import get_llf
from src.utils.consts import (
    MAD_FOLDER,
    MAD_LFF_OUTPUT_FOLDER,
    MPTRAJ_FOLDER,
    MPTRAJ_LFF_OUTPUT_FOLDER,
    SPLITS,
    SUBSETS,
    TYPE,
)


def save_llfs(output_path, values):
    folder_path = os.path.dirname(output_path)
    os.makedirs(folder_path, exist_ok=True)

    torch.save(values, output_path)
    print(f"Saved to {output_path}")


def get_mad_llfs(model):

    os.makedirs(MAD_LFF_OUTPUT_FOLDER, exist_ok=True)

    for split in SPLITS:
        print(f"Processing {split}...")

        for subset in SUBSETS:
            subset_input_path = os.path.join(MAD_FOLDER, subset, TYPE, f"{split}.xyz")

            mean_llfs_tensor, std_llfs_tensor, combined_feats = get_llf(
                model, subset_input_path
            )

            base_folder = os.path.join(MAD_LFF_OUTPUT_FOLDER, subset)
            save_llfs(
                os.path.join(base_folder, "mean", f"{split}.pt"), mean_llfs_tensor
            )
            save_llfs(os.path.join(base_folder, "std", f"{split}.pt"), std_llfs_tensor)
            save_llfs(
                os.path.join(base_folder, "combined", f"{split}.pt"), combined_feats
            )


def get_mptraj_llfs(model):
    os.makedirs(MPTRAJ_LFF_OUTPUT_FOLDER, exist_ok=True)

    datasets = [d for d in os.listdir(MPTRAJ_FOLDER)]
    datasets.sort()

    for dataset in datasets:
        dataset_name = dataset.split(".")[0]
        dataset_input_path = os.path.join(MPTRAJ_FOLDER, dataset)

        llfs_tensor = get_llf(model, dataset_input_path)

        subset_output_path = os.path.join(
            MPTRAJ_LFF_OUTPUT_FOLDER, f"{dataset_name}.pt"
        )
        torch.save(llfs_tensor, subset_output_path)
        print(f"Saved to {subset_output_path}")
