from ase.io import read, write

import os
import torch

from tqdm import tqdm

from metatrain.utils.data import read_systems
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatensor.torch.atomistic import ModelOutput

from src.utils.consts import SPLITS, BATCH_SIZE, DEVICE


def merge_subsets(dataset_folder, output_folder, dataset_type):
    os.makedirs(output_folder, exist_ok=True)

    for split in SPLITS:
        output_path = os.path.join(output_folder, f"{split}.xyz")
        all_atoms = []

        for root, _, _files in os.walk(dataset_folder):
            target_file = os.path.join(root, dataset_type, f"{split}.xyz")

            if os.path.exists(target_file):
                atoms_list = read(target_file, index=":")
                all_atoms.extend(atoms_list)

        write(output_path, all_atoms, format="xyz")


def get_llf(model, dataset_path):
    systems = read_systems(dataset_path)

    neighbor_list_options = model.requested_neighbor_lists()

    outputs = {
        "energy": ModelOutput(per_atom=False),
        "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=False),
    }

    all_last_layer_features = []

    for i in tqdm(range(0, len(systems), BATCH_SIZE)):
        batch_systems = systems[i : i + BATCH_SIZE]

        processed_batch = []
        for system in batch_systems:
            system_with_nl = get_system_with_neighbor_lists(
                system, neighbor_list_options
            )
            processed_batch.append(
                system_with_nl.to(dtype=torch.float32, device=DEVICE)
            )

        with torch.no_grad():
            batch_predictions = model(processed_batch, outputs)
            batch_features = (
                batch_predictions["mtt::aux::energy_last_layer_features"].block().values
            )

        all_last_layer_features.append(batch_features)

    last_layer_features = torch.cat(all_last_layer_features, dim=0)

    return last_layer_features
