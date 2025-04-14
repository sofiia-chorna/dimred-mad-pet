import os

import click
import numpy as np
import torch

from metatrain.experimental.nativepet import NativePET
from src.get_llfs import run_get_llfs
from src.train.predict import run_predict
from src.train.train import load_subset_models, save_subset_models, train_model
from src.utils.consts import (
    DATASET_FOLDER,
    DEVICE,
    LFF_OUTPUT_FOLDER,
    PROJECTION_FOLDER,
    SUBSETS,
    TYPE,
)
from src.utils.file import load_txt
from src.utils.plot import plot_split_comparison

# from src.dimred import run_pca


@click.group()
def main():
    pass


@main.command()
def get_llfs():
    model = NativePET.load_checkpoint("pet-mad-latest.ckpt").eval().to(DEVICE)

    run_get_llfs(model, DATASET_FOLDER, LFF_OUTPUT_FOLDER, TYPE)


# @main.command()
# def get_pca():
#    run_pca(LFF_OUTPUT_FOLDER)


@main.command()
def train():
    # load train llfs
    train_features = {}
    for subset_name in SUBSETS:
        path = os.path.join(LFF_OUTPUT_FOLDER, subset_name, "train.dat")
        train_features[subset_name] = np.loadtxt(path)

    # load projections
    smap_train_actual = load_txt(PROJECTION_FOLDER, "train")

    subset_models = {}
    for subset_name in SUBSETS:
        X = torch.tensor(train_features[subset_name], dtype=torch.float32)
        Y = torch.tensor(smap_train_actual[subset_name][:, :2], dtype=torch.float32)

        subset_models[subset_name] = train_model(X, Y, input_dim=X.shape[1])

    save_subset_models(subset_models)


@main.command()
def predict():
    subset_models_loaded = load_subset_models(SUBSETS)

    smap_test_actual = load_txt(PROJECTION_FOLDER, "test")

    pred = run_predict(subset_models_loaded, "test")
    actual = {k: v[:, :2] for k, v in smap_test_actual.items()}

    plot_split_comparison(actual, pred, "mlp")


if __name__ == "__main__":
    main()
