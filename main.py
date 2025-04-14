import os

import click
import numpy as np

from metatrain.experimental.nativepet import NativePET
from src.get_llfs import run_get_llfs
from src.train.predict import run_predict
from src.train.run_train import run_train
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

    run_train(train_features, smap_train_actual)


@main.command()
def predict():
    smap_test_actual = load_txt(PROJECTION_FOLDER, "test")

    pred = run_predict("test")
    actual = {k: v[:, :2] for k, v in smap_test_actual.items()}

    plot_split_comparison(actual, pred, "mlp")


if __name__ == "__main__":
    main()
