import numpy as np
import metatensor
import torch
from metatensor.torch import (
    Labels,
    TensorBlock,
    TensorMap,
    mean_over_samples,
    # std_over_samples,
)
from metatensor.torch.atomistic import ModelOutput
from tqdm import tqdm

from metatrain.utils.data import read_systems
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from src.utils.consts import BATCH_SIZE, DEVICE, SPLITS


def std_over_samples(tensor_map, sample_names):
    device = block.values.device

    new_blocks = []
    for key in tensor_map.keys:
        block = tensor_map.block(key)
        original_samples = block.samples
        sample_labels = original_samples.names

        group_by = [name for name in sample_labels if name not in sample_names]

        if not group_by:
            # single group case
            group_indices = [torch.arange(len(original_samples), device=device)]
            new_sample_values = torch.zeros((1, 0), dtype=torch.int32, device=device)
            new_sample_names = []
        else:
            # create group keys using native python integers
            group_keys = [
                tuple(original_samples[i][name] for name in group_by)
                for i in range(len(original_samples))
            ]

            unique_groups = sorted(set(group_keys))
            group_indices = []
            for group in unique_groups:
                indices = [i for i, gk in enumerate(group_keys) if gk == group]
                group_indices.append(
                    torch.tensor(indices, dtype=torch.long, device=device)
                )

            new_sample_values = torch.tensor(
                [list(group) for group in unique_groups],
                dtype=torch.int32,
                device=device,
            )
            new_sample_names = group_by

        # compute std for each group
        new_values = []
        for indices in group_indices:
            group_data = block.values[indices]
            if group_data.shape[0] < 2:
                group_std = torch.zeros(
                    group_data.shape[1:],
                    dtype=group_data.dtype,
                    device=group_data.device,
                )
            else:
                group_std = torch.std(group_data, dim=0, unbiased=True)
            new_values.append(group_std)

        new_values = torch.stack(new_values, dim=0)

        new_samples = Labels(names=new_sample_names, values=new_sample_values)
        new_block = TensorBlock(
            values=new_values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return TensorMap(tensor_map.keys, new_blocks)


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

    # mean features
    mean_features = mean_over_samples(tensormap, sample_names=["atom"])
    mean_tensor_values_list = [block.values for block in mean_features.blocks()]
    mean_torch_tensor_values = torch.cat(mean_tensor_values_list, dim=0)

    # std features
    std_features = std_over_samples(tensormap, sample_names=["atom"])
    std_tensor_values_list = [block.values for block in std_features.blocks()]
    std_torch_tensor_values = torch.cat(std_tensor_values_list, dim=0)
    # std_torch_tensor_values = torch.nan_to_num(std_torch_tensor_values, nan=0.0)

    std_tensor_values_list = []
    for j, block in enumerate(std_features.blocks()):
        values = block.values

        if torch.isnan(values).any():
            print(f"\n=== NaN Debugging for Block {j} ===")

            # 1. Check input statistics
            original_block = all_blocks[j]
            print("Original values stats:")
            print(f"  Shape: {original_block.values.shape}")
            print(f"  Min: {original_block.values.min().item():.4f}")
            print(f"  Max: {original_block.values.max().item():.4f}")
            print(f"  Mean: {original_block.values.mean().item():.4f}")

            # 2. Check for Infs
            if torch.isinf(original_block.values).any():
                print("  Contains INF values!")

            # 3. Check variance before std
            variance = torch.var(original_block.values, dim=0, unbiased=True)
            print("Variance stats:")
            print(f"  Min variance: {variance.min().item():.4e}")
            print(f"  Max variance: {variance.max().item():.4e}")
            print(f"  Zero variances: {(variance == 0).sum().item()}")

            # 4. Check specific problematic features
            nan_mask = torch.isnan(values)
            nan_indices = torch.where(nan_mask)
            print(f"NaN positions: {list(zip(*nan_indices))}")

            # Show problematic features
            for sample_idx, feat_idx in zip(*nan_indices):
                feat_values = original_block.values[:, feat_idx]
                print(f"\nFeature {feat_idx} in sample {sample_idx}:")
                print(f"  Values: {feat_values.cpu().numpy()}")
                print(f"  Variance: {feat_values.var(unbiased=True).item():.4e}")
                print(f"  All zeros: {torch.all(feat_values == 0)}")

        if values.shape[0] < 2:
            print(f"Warning: std_features block {j} has < 2 samples, setting to zeros")
            values = torch.zeros_like(values)

        elif torch.isnan(values).any():
            nan_count = torch.isnan(values).sum().item()
            print(f"NaNs in std_features block {j}, count: {nan_count}")
            values = torch.where(torch.isnan(values), torch.zeros_like(values), values)

        new_block = TensorBlock(
            values=values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        if block.has_gradient("positions"):
            new_block.add_gradient("positions", block.gradient("positions"))
        std_tensor_values_list.append(new_block.values)

    std_torch_tensor_values = torch.cat(std_tensor_values_list, dim=0)

    if torch.isnan(std_torch_tensor_values).any():
        print(
            f"NaNs detected in final std_torch_tensor_values, count: {torch.isnan(std_torch_tensor_values).sum().item()}"
        )

    # combine
    combined_features = torch.cat(
        [mean_torch_tensor_values, std_torch_tensor_values], dim=1
    )

    return mean_torch_tensor_values, std_torch_tensor_values, combined_features
