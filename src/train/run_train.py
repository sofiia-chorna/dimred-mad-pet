import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils.consts import DEVICE, SUBSETS
from src.train.predictor import MultiSubsetModel


def run_train(train_features, smap_train_actual):
    # Prepare data for each subset
    subset_data = {}
    subset_dims = {}
    train_datasets = {}
    val_datasets = {}

    for subset_name in SUBSETS:
        X = train_features[subset_name]
        Y = smap_train_actual[subset_name][:, :2]

        X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # Standardize features
        X_mean = X_train_np.mean(axis=0)
        X_std = X_train_np.std(axis=0) + 1e-8
        X_train_std = (X_train_np - X_mean) / X_std
        X_val_std = (X_val_np - X_mean) / X_std

        # Fit Ridge model
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_std, Y_train_np)
        lin_pred_train = ridge.predict(X_train_std)
        lin_pred_val = ridge.predict(X_val_std)

        # Compute residuals
        residual_train = Y_train_np - lin_pred_train
        residual_val = Y_val_np - lin_pred_val

        # Standardize residuals
        y_mean = residual_train.mean(axis=0)
        y_std = residual_train.std(axis=0) + 1e-8
        y_train_std = (residual_train - y_mean) / y_std
        y_val_std = (residual_val - y_mean) / y_std

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_std, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_std, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_std, dtype=torch.float32)

        # Store subset data
        subset_data[subset_name] = {
            "ridge": ridge,
            "X_mean": X_mean,
            "X_std": X_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }
        subset_dims[subset_name] = X.shape[1]

        # Create datasets
        train_datasets[subset_name] = TensorDataset(X_train_tensor, y_train_tensor)
        val_datasets[subset_name] = TensorDataset(X_val_tensor, y_val_tensor)

    # Initialize the multi-subset model
    model = MultiSubsetModel(subset_dims=subset_dims)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )

    # Training loop
    epochs = 1000
    patience = 100
    best_val_loss = float("inf")
    patience_counter = 0
    batch_size = 64

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        # Train on each subset
        for subset_name in SUBSETS:
            train_loader = DataLoader(
                train_datasets[subset_name], batch_size=batch_size, shuffle=True
            )
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X, subset_name)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                total_train_samples += batch_X.size(0)

        train_loss /= total_train_samples

        # Validation
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for subset_name in SUBSETS:
                val_loader = DataLoader(
                    val_datasets[subset_name], batch_size=batch_size
                )
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_X, subset_name)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    total_val_samples += batch_X.size(0)

        val_loss /= total_val_samples
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save temporary best model state
            torch.save(model.state_dict(), "temp_best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

    # Load best model state
    model.load_state_dict(torch.load("temp_best_model.pth"))

    # Save single checkpoint
    checkpoint = {"model_state_dict": model.state_dict(), "subset_data": subset_data}
    save_path = os.path.join("saved_models", "checkpoint.pth")
    os.makedirs("saved_models", exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
