import os

import torch

from src.utils.consts import LLFS_OUTPUT_FOLDER


def predict(subset_models_dict, split="test"):
    res = {}

    for subset_name, output in subset_models_dict.items():
        mlp_model = output.mlp_model
        lin_model = output.lin_model

        x_path = os.path.join(LLFS_OUTPUT_FOLDER, subset_name, f"{split}.pt")

        X = torch.load(x_path, map_location="cpu").float()
        X_np = X.numpy()

        # standardize
        X_std_np = (X_np - output.X_mean) / output.X_std
        X_std_tensor = torch.tensor(X_std_np, dtype=torch.float32)

        # predict
        with torch.no_grad():
            mlp_model.eval()
            mlp_pred_std = mlp_model(X_std_tensor).numpy()

        # unstandardize
        mlp_pred = mlp_pred_std * output.y_std.numpy() + output.y_mean.numpy()
        lin_pred = lin_model.predict(X_std_np)
        res[subset_name] = lin_pred + mlp_pred

    return res
