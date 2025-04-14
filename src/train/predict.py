import os

import torch

from src.utils.consts import LFF_OUTPUT_FOLDER, DEVICE


def run_predict(subset_models_dict, split="test"):
    res = {}

    for subset_name, items in subset_models_dict.items():
        (
            mlp_model,
            lin_model,
            X_mean,
            X_std,
            y_mean,
            y_std,
        ) = items

        x_path = os.path.join(LFF_OUTPUT_FOLDER, subset_name, f"{split}.pt")
        X = torch.load(x_path, map_location="cpu", weights_only=False).float()

        X_np = X.numpy()

        # standardize
        X_std_np = (X_np - X_mean) / X_std
        X_std_tensor = torch.tensor(X_std_np, dtype=torch.float32)

        # predict
        with torch.no_grad():
            mlp_model.eval()
            mlp_pred_std = mlp_model(X_std_tensor).numpy()

        # unstandardize
        mlp_pred = mlp_pred_std * y_std.cpu().numpy() + y_mean.cpu().numpy()
        lin_pred = lin_model.predict(X_std_np)
        res[subset_name] = lin_pred + mlp_pred

    return res
