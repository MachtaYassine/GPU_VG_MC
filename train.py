import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ----------------------------
# Load Dataset
# ----------------------------

def load_dataset(folder):
    all_files = list(Path(folder).glob("*.csv"))
    print(f"Found {len(all_files)} files in {folder}")
    data = []
    for file in all_files:
        name = file.stem  # e.g., VG_T0.25_K0.9330
        try:
            parts = name.split('_')
            T_val = float(parts[1][1:])  # strip 'T'
            K_val = float(parts[2][1:])  # strip 'K'
        except Exception as e:
            print(f"Failed to parse: {name} — {e}")
            continue

        df = pd.read_csv(file)
        df['T'] = T_val
        df['K'] = K_val
        data.append(df)
    if not data:
        return pd.DataFrame()
    return pd.concat(data, ignore_index=True)

# ----------------------------
# Dataloaders
# ----------------------------

def to_loader(X, y,device, batch_size=64, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

# ----------------------------
# Model Definition
# ----------------------------

class VGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Training Loop
# ----------------------------

def train(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_losses.append(train_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
                val_loss += loss.item() * len(X_batch)

        val_losses.append(val_loss / len(val_loader.dataset))

        print(f"Epoch {epoch:3d} | Train Loss: {train_losses[-1]:.5f} | Val Loss: {val_losses[-1]:.5f}")

    # save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "checkpoints/vgnet_model.pth")

    return train_losses, val_losses

# ----------------------------
# Testing Loop
# ----------------------------

def test(model, test_loader):
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)
            predictions.append(pred)
            true_values.append(y_batch)

    predictions = torch.cat(predictions)
    true_values = torch.cat(true_values)

    mse = ((predictions - true_values) ** 2).mean()
    print(f"Test MSE: {mse:.5f}")

    return predictions, true_values

# ----------------------------
# Evaluate the model from the checkpoint
# ----------------------------

def eval_state(path, X_test, y_test, df_test, model_class, input_dim, output_dim, device):
    checkpoint = torch.load(path, map_location=device)
    model = model_class().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor).squeeze().cpu().numpy()
    targets = y_tensor.cpu().numpy()

    errors = df_test[" 95cI"].to_numpy() if " 95cI" in df_test.columns else np.full_like(targets, 0.01)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(targets, preds, s=2, alpha=0.3)
    axs[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axs[0].set_title("Predicted vs True Prices")
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Predicted")

    # Metrics
    rho = np.corrcoef(targets, preds)[0, 1]
    r2 = 1 - np.sum((preds - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    acc = np.mean((targets - errors <= preds) & (preds <= targets + errors))
    acc_relaxed = np.mean((targets - 2 * errors <= preds) & (preds <= targets + 2 * errors))

    axs[1].hist(targets - preds, bins=100, alpha=0.7)
    axs[1].set_title("Prediction Error Histogram")

    plt.tight_layout()
    plt.show()

    print(f"Linear corr (ρ): {rho:.4f}, R²: {r2:.4f}")
    print(f"Accuracy (±CI): {acc:.4f}, Relaxed Accuracy (±2×CI): {acc_relaxed:.4f}")

    return preds

# ----------------------------
# Monotonicity and Convexity Analysis
# ----------------------------

def analyze_monotonicity_convexity(model, df_test, features, target_col, device, verbose=False, plot=True):
    model.eval()
    T_values = np.sort(df_test['T'].unique())
    K_values = np.sort(df_test['K'].unique())

    # Randomly pick 6 configurations of theta, kappa, sigma, T
    unique_configs = df_test.groupby(['theta', 'kappa', 'sigma', 'T']).size().reset_index().drop(0, axis=1)
    selected_configs = unique_configs.sample(n=6, random_state=42)

    results = []

    # For each value of K, check the prices for the selected configurations
    for K in K_values:
        config_data = pd.DataFrame()
        for _, config in selected_configs.iterrows():
            # Filter data for the current configuration and K value
            filtered_data = df_test[
                (df_test['theta'] == config['theta']) &
                (df_test['kappa'] == config['kappa']) &
                (df_test['sigma'] == config['sigma']) &
                (df_test['T'] == config['T']) &
                (df_test['K'] == K)
            ]
            config_data = pd.concat([config_data, filtered_data])

        # Prepare data for model inference
        X = config_data[features].values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # Perform model inference
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()

        # Store results
        config_data['predicted_price'] = predictions
        results.append(config_data)

    # Concatenate all results
    results_df = pd.concat(results)

    # Group by configuration and plot the prices according to different K_values
    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (_, config) in enumerate(selected_configs.iterrows()):
            config_results = results_df[
                (results_df['theta'] == config['theta']) &
                (results_df['kappa'] == config['kappa']) &
                (results_df['sigma'] == config['sigma']) &
                (results_df['T'] == config['T'])
            ]
            K_vals = config_results['K'].values
            pred_prices = config_results['predicted_price'].values

            # Compute first derivative
            first_derivative = np.gradient(pred_prices, K_vals)

            # Compute second derivative
            second_derivative = np.gradient(first_derivative, K_vals)

            # Plot predicted and real prices
            axes[i].plot(K_vals, pred_prices, marker='o', label='Predicted Price')
            axes[i].plot(K_vals, config_results[target_col], marker='x', label='Real Price', linestyle='--')
            axes[i].set_xlabel('K')
            axes[i].set_ylabel('Price')
            axes[i].set_title(f'Price vs K for Config {i+1}')
            axes[i].legend(loc='upper left')

            # Plot second derivative
            ax2 = axes[i].twinx()
            ax2.plot(K_vals, second_derivative, color='green', linestyle='-.', label='Second Derivative', alpha=0.6, linewidth=1)
            ax2.set_ylabel('Second Derivative')
            ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    # Randomly pick 6 configurations of theta, kappa, sigma,K
    unique_configs = df_test.groupby(['theta', 'kappa', 'sigma', 'K']).size().reset_index().drop(0, axis=1)
    selected_configs = unique_configs.sample(n=6, random_state=42)

    results = []
    print(f'length of T_values: {len(T_values)}')
    # For each value of K, check the prices for the selected configurations
    for T in T_values:
        config_data = pd.DataFrame()
        for _, config in selected_configs.iterrows():
            # Filter data for the current configuration and K value
            filtered_data = df_test[
                (df_test['theta'] == config['theta']) &
                (df_test['kappa'] == config['kappa']) &
                (df_test['sigma'] == config['sigma']) &
                (df_test['K'] == config['K']) &
                (df_test['T'] == T)
            ]
            config_data = pd.concat([config_data, filtered_data])

        # Prepare data for model inference
        X = config_data[features].values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # Perform model inference
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()

        # Store results
        config_data['predicted_price'] = predictions
        results.append(config_data)

    # Concatenate all results
    results_df = pd.concat(results)

    # Group by configuration and plot the prices according to different K_values
    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (_, config) in enumerate(selected_configs.iterrows()):
            config_results = results_df[
                (results_df['theta'] == config['theta']) &
                (results_df['kappa'] == config['kappa']) &
                (results_df['sigma'] == config['sigma']) 
            ]
            T_vals = config_results['T'].values
            print(f'length of T_vals: {len(T_vals)}')
            pred_prices = config_results['predicted_price'].values


            # Plot predicted and real prices
            axes[i].plot(T_vals, pred_prices, marker='o', label='Predicted Price')
            axes[i].plot(T_vals, config_results[target_col], marker='x', label='Real Price', linestyle='--')
            axes[i].set_xlabel('T')
            axes[i].set_ylabel('Price')
            axes[i].set_title(f'Price vs T for Config {i+1}')
            axes[i].legend(loc='upper left')

            

        plt.tight_layout()
        plt.show()
    if verbose:
        print("Analysis complete.")
        
    return
    



# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a VGNet model.")
    parser.add_argument("--train_folder", type=str, default="Training/", help="Path to the training data folder.")
    parser.add_argument("--test_folder", type=str, default="Testing/", help="Path to the testing data folder.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training and validation.")
    parser.add_argument("--test_batch_size", type=int, default=512, help="Batch size for testing.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/vgnet_model.pth", help="Path to the model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training and evaluation (cpu or cuda).")

    args = parser.parse_args()

    print("Loading training data...")
    df_train_full = load_dataset(args.train_folder)
    if df_train_full.empty:
        raise ValueError("No training data loaded. Check your 'Training/' directory and file format.")

    print("Splitting training and validation sets...")
    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=42)

    features = ['kappa', 'theta', 'sigma', 'T', 'K']
    target = 'price'

    X_train = df_train[features].values
    y_train = df_train[target].values
    X_val = df_val[features].values
    y_val = df_val[target].values

    train_loader = to_loader(X_train, y_train,args.device , batch_size=args.batch_size)
    val_loader = to_loader(X_val, y_val,args.device , batch_size=args.batch_size, shuffle=False)

    print("Initializing model...")
    model = VGNet().to(args.device)
    

    print("Starting training...")
    train_losses, val_losses = train(model, train_loader, val_loader, epochs=args.epochs)

    # ----------------------------
    # Plotting the loss curves
    # ----------------------------

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='black')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='blue')
    plt.title('Validation and Train Loss Over Epochs')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('VG_loss_curve.png')
    plt.show()

    # ----------------------------
    # Test the model and get the predicted prices
    # ----------------------------

    print("Loading test data...")
    df_test = load_dataset(args.test_folder)
    if df_test.empty:
        raise ValueError("No test data found in 'Testing/' directory.")

    X_test = df_test[features].values
    y_test = df_test[target].values
    test_loader = to_loader(X_test, y_test,args.device, batch_size=args.test_batch_size, shuffle=False)

    predictions, true_values = test(model, test_loader)

    # Print first 10 predictions
    print("Predicted prices:", predictions[:10])
    print("True prices:     ", true_values[:10])

    # ----------------------------
    # Evaluate the best model from a checkpoint
    # ----------------------------
    eval_state(args.checkpoint_path, X_test, y_test, df_test, VGNet, 5, 1, torch.device(args.device))

    # ----------------------------
    # Analyze monotonicity and convexity
    # ----------------------------
    analyze_monotonicity_convexity(
    model=model,
    df_test=df_test,
    features=features,
    target_col='price',
    device=args.device,
    verbose=True,
    plot=True
)

