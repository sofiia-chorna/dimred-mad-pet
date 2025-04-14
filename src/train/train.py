import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.train.mlp import Predictor
from src.utils.consts import DEVICE, LFF_OUTPUT_FOLDER


def run_train_loop(
    model, train_loader, val_loader, lr=0.001, epochs=1000, patience=100
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # criterion = nn.L1Loss()
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                outputs = model(batch_X)

                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")


def train_model(X, Y, input_dim, batch_size=64):
    X_np = X.numpy()
    Y_np = Y.numpy()

    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
        X_np, Y_np, test_size=0.2, random_state=42
    )

    # standardize
    X_mean = X_train_np.mean(axis=0)
    X_std = X_train_np.std(axis=0) + 1e-8
    X_train_std = (X_train_np - X_mean) / X_std
    X_val_std = (X_val_np - X_mean) / X_std

    # fit ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_std, Y_train_np)
    lin_pred_train = ridge.predict(X_train_std)
    lin_pred_val = ridge.predict(X_val_std)

    # residuals
    y_non_linear_train = torch.tensor(Y_train_np - lin_pred_train, dtype=torch.float32)
    y_non_linear_train = y_non_linear_train.to(DEVICE)
    y_non_linear_val = torch.tensor(Y_val_np - lin_pred_val, dtype=torch.float32)
    y_non_linear_val = y_non_linear_val.to(DEVICE)

    # standardize
    y_mean = y_non_linear_train.mean(dim=0)
    y_std = y_non_linear_train.std(dim=0) + 1e-8
    y_train_std = (y_non_linear_train - y_mean) / y_std
    y_val_std = (y_non_linear_val - y_mean) / y_std

    X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_std, dtype=torch.float32).to(DEVICE)

    # dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_std)
    val_dataset = TensorDataset(X_val_tensor, y_val_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = Predictor(input_dim=input_dim)
    model.to(DEVICE)

    run_train_loop(model, train_loader, val_loader)

    # load best model
    model.load_state_dict(torch.load("best_model.pth"))
    return model, ridge, X_mean, X_std, y_mean, y_std


def save_subset_models(subset_models, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    for subset_name, items in subset_models.items():
        (mlp_model, lin_model, X_mean, X_std, y_mean, y_std) = items

        mlp_path = os.path.join(save_dir, f"{subset_name}_mlp.pth")
        torch.save(mlp_model.state_dict(), mlp_path)

        with open(os.path.join(save_dir, f"{subset_name}_ridge.pkl"), "wb") as f:
            pickle.dump(
                {
                    "ridge": lin_model,
                    "X_mean": X_mean,
                    "X_std": X_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                },
                f,
            )


def load_subset_models(subsets, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    loaded_models = {}

    for subset_name in subsets:
        X = torch.load(os.path.join(LFF_OUTPUT_FOLDER, subset_name, "train.pt")).float()

        mlp_model = Predictor(input_dim=X.shape[1])
        mlp_model.load_state_dict(
            torch.load(os.path.join(save_dir, f"{subset_name}_mlp.pth"))
        )

        with open(os.path.join(save_dir, f"{subset_name}_ridge.pkl"), "rb") as f:
            ridge_data = pickle.load(f)

        loaded_models[subset_name] = (
            mlp_model,
            ridge_data["ridge"],
            ridge_data["X_mean"],
            ridge_data["X_std"],
            ridge_data["y_mean"],
            ridge_data["y_std"],
        )

    return loaded_models
