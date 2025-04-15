import os

import numpy as np
import torch

from src.train.mlp import MLP
from src.utils.consts import DEVICE, MAD_LFF_OUTPUT_FOLDER, SUBSETS


def run_predict(split="test"):
    checkpoint_path = os.path.join("saved_models", "checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    subset_dims = {}
    for subset in SUBSETS:
        train_file = os.path.join(MAD_LFF_OUTPUT_FOLDER, subset, "train.dat")
        subset_dims[subset] = np.loadtxt(train_file).shape[1]

    model = MLP(subset_dims=subset_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    subset_data = checkpoint["subset_data"]

    res = {}
    for subset_name in SUBSETS:
        data = subset_data[subset_name]
        ridge = data["ridge"]
        X_mean = data["X_mean"]
        X_std = data["X_std"]
        y_mean = data["y_mean"]
        y_std = data["y_std"]

        x_path = os.path.join(MAD_LFF_OUTPUT_FOLDER, subset_name, f"{split}.pt")
        X = torch.load(x_path, map_location="cpu", weights_only=False).float().numpy()

        # standardize
        X_std_np = (X - X_mean) / X_std
        X_std_tensor = torch.tensor(X_std_np, dtype=torch.float32).to(DEVICE)

        # predict
        with torch.no_grad():
            mlp_pred_std = model(X_std_tensor, subset_name).cpu().numpy()

        # unstandardize
        mlp_pred = mlp_pred_std * y_std + y_mean
        lin_pred = ridge.predict(X_std_np)
        res[subset_name] = lin_pred + mlp_pred

    return res
