from ase.io import read, write
import os
import torch

from tqdm import tqdm


import metatensor
from metatensor.torch import TensorMap, Labels
from metatensor.torch.atomistic import ModelOutput

from metatrain.utils.data import read_systems
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

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
        "energy": ModelOutput(per_atom=True),
        "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=True),
    }

    all_blocks = []

    for i in tqdm(range(0, len(systems), BATCH_SIZE)):
        batch_systems = systems[i : i + BATCH_SIZE]
        processed_batch = [
            get_system_with_neighbor_lists(system, neighbor_list_options).to(
                dtype=torch.float32, device=DEVICE
            )
            for system in batch_systems
        ]

        with torch.no_grad():
            batch_predictions = model(processed_batch, outputs)
            block = batch_predictions["mtt::aux::energy_last_layer_features"].block()
            all_blocks.append(block)

    keys = Labels(
        names=["batch"],
        values=torch.tensor(
            [[i] for i in range(len(all_blocks))], dtype=torch.int32, device=DEVICE
        ),
    )
    tensormap = TensorMap(keys=keys, blocks=all_blocks)

    mean_features = metatensor.torch.mean_over_samples(tensormap, sample_names=["atom"])

    tensor_values_list = [block.values for block in mean_features.blocks()]
    torch_tensor_values = torch.cat(tensor_values_list, dim=0)

    return torch_tensor_values
