import ase.io
import os

from tqdm import tqdm
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.consts import SUBSETS, LFF_OUTPUT_FOLDER, DATASET_FOLDER, TYPE
from src.evaluation.evaluation import eval


def get_concat_llfs(split):
    features = []

    for subset_name in SUBSETS:
        features.append(
            torch.load(
                f"{LFF_OUTPUT_FOLDER}/{subset_name}/{split}.pt", weights_only=False
            )
        )

    return torch.cat(features, dim=0)


def get_pca(split, features, pca):
    transformed = pca.transform(features)

    subset_results = {}
    current_idx = 0

    for subset_name in tqdm(SUBSETS, desc="Processing subsets"):
        subset_dir = os.path.join(DATASET_FOLDER, subset_name)

        if not os.path.isdir(subset_dir):
            continue

        xyz_path = os.path.join(subset_dir, f"{split}.xyz")
        if not os.path.exists(xyz_path):
            continue

        atoms = ase.io.read(xyz_path, ":")
        n_structures = len(atoms)

        subset_results[subset_name] = transformed[
            current_idx : current_idx + n_structures
        ]
        current_idx += n_structures

    if current_idx != len(features):
        raise ValueError(
            f"Feature count mismatch: Processed {current_idx} features, "
            f"but had {len(features)} total"
        )

    return subset_results


def predict(split, scaler, subset_weights):
    predictions = {}

    total_processed = 0

    for subset_name in tqdm(subset_weights.keys(), desc=f"Predicting {split}"):
        llfs_path = os.path.join(LFF_OUTPUT_FOLDER, subset_name, f"{split}.pt")

        if not os.path.exists(llfs_path):
            continue

        subset_features = torch.load(llfs_path, weights_only=False).cpu().numpy()
        n_structures = subset_features.shape[0]
        total_processed += n_structures

        scaled_features = scaler.transform(subset_features)
        W = subset_weights[subset_name].numpy()
        pca_pred = scaled_features @ W

        predictions[subset_name] = pca_pred

    print(
        f"Predicted PCA for {total_processed} structures across {len(predictions)} subsets"
    )
    return predictions


def run_pca(split):
    train_features = get_concat_llfs("train")

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    split_llfs = get_concat_llfs(split).cpu()
    split_features_scaled = scaler.transform(split_llfs)

    pca = PCA(n_components=2)
    pca.fit(train_features_scaled)
    print(
        "explained variance ratio for scaled features:", pca.explained_variance_ratio_
    )

    train_features_np = train_features_scaled.numpy()
    train_subset_results = get_pca("train", train_features_np, pca)

    # fit regression weights
    subset_weights = {}
    for subset_name, subset_pca_values in train_subset_results.items():
        subset_features = train_features_scaled[: len(subset_pca_values)]
        train_features_scaled = train_features_scaled[len(subset_pca_values) :]
        W = torch.linalg.lstsq(
            torch.tensor(subset_features, dtype=torch.float32),
            torch.tensor(subset_pca_values, dtype=torch.float32),
        ).solution
        subset_weights[subset_name] = W

    predicted_pca_test = predict(split, scaler, subset_weights)
    actual_pca_test = get_pca(split, split_features_scaled, pca)

    eval_test = eval(predicted_pca_test, actual_pca_test)
    print(eval_test)
