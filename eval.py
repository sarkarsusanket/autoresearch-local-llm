import argparse
from datetime import datetime
import os
import sys
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Force PyTorch to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset_data(dataset_name):
    """Dynamically loads datasets based on name and infers target columns"""
    paths = {
        "health": r"E:\Data\PDFM\medical.csv",
        "climate": r"E:\Data\RetreivalTasks\climate-justice\usa\climate.csv",
        "fema": r"E:\Data\RetreivalTasks\FEMA\fema.csv",
        "svi": r"E:\Data\RetreivalTasks\SVI_2022_US\svi.csv",
    }

    if dataset_name not in paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    path = paths[dataset_name]
    if not os.path.exists(path):
        print(f"Warning: Path {path} not found. Skipping {dataset_name}...")
        return None, []

    df = pd.read_csv(path)

    lon_col, lat_col = None, None
    for lon in ["longitude", "lon", "lng", "long"]:
        if lon in df.columns:
            lon_col = lon
    for lat in ["latitude", "lat"]:
        if lat in df.columns:
            lat_col = lat

    if lon_col and lat_col:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )
        if dataset_name == "health":
            target_cols = [col for col in df.columns if col.startswith("Percent_Person_")]
        elif dataset_name in ["climate", "fema"]:
            target_cols = [col for col in df.columns if col.endswith("SCORE") or col.endswith("RISKS")]
        elif dataset_name == "svi":
            target_cols = [col for col in df.columns[:-2]]
    else:
        coords = df.iloc[:, -2:]
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(coords.iloc[:, 0], coords.iloc[:, 1]),
            crs="EPSG:4326",
        )
        target_cols = df.columns[:-2].tolist()

    return gdf, target_cols


def get_spatial_embeddings(data_gdf, emb_gdf):
    """Maps geometries using a pre-loaded embeddings GeoDataFrame"""
    if data_gdf.crs != emb_gdf.crs:
        data_gdf = data_gdf.to_crs(emb_gdf.crs)

    emb_cols = [c for c in emb_gdf.columns if c.startswith("emb_")]

    # Spatial join optimization
    joined = gpd.sjoin(
        data_gdf,
        emb_gdf[["geometry"] + emb_cols],
        how="left",
        predicate="within",
    )

    valid_joined = joined.dropna(subset=emb_cols)
    X = valid_joined[emb_cols].values
    return X, valid_joined


class MultiTaskMLP(nn.Module):
    """A small, highly efficient MLP that predicts all targets simultaneously"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim) # Outputs raw logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)


def process_all_targets_mlp(X, Y_df, target_cols, n_splits=5, epochs=15, batch_size=256):
    """
    Trains a multi-task neural network on all targets simultaneously using PyTorch GPU acceleration.
    """
    # Preprocess targets matrix (Binarize continuous targets around median)
    Y_np = np.zeros((len(Y_df), len(target_cols)), dtype=np.float32)
    valid_mask = np.ones(len(Y_df), dtype=bool)

    for i, col in enumerate(target_cols):
        vals = Y_df[col].values
        col_mask = ~pd.isna(vals)
        valid_mask &= col_mask  # Keep track of completely clean rows
        
        # Binarize
        if len(np.unique(vals[col_mask])) > 2:
            median_val = np.nanmedian(vals)
            Y_np[:, i] = (vals >= median_val).astype(np.float32)
        else:
            Y_np[:, i] = np.nan_to_num(vals, nan=0.0).astype(np.float32)

    # Filter out rows with missing targets
    X_clean = X[valid_mask]
    Y_clean = Y_np[valid_mask]

    if len(X_clean) < max(10, n_splits) or Y_clean.shape[1] == 0:
        return {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store dynamic macro F1 tracking matrices [folds, targets]
    fold_f1_matrix = np.zeros((n_splits, len(target_cols)))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_clean)):
        X_train, Y_train = torch.tensor(X_clean[train_idx], dtype=torch.float32), torch.tensor(Y_clean[train_idx], dtype=torch.float32)
        X_test, Y_test = torch.tensor(X_clean[test_idx], dtype=torch.float32), torch.tensor(Y_clean[test_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

        # Initialize network
        model = MultiTaskMLP(X_clean.shape[1], Y_clean.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        # Quick neural net training loop
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate Multi-task targets
        model.eval()
        with torch.no_grad():
            test_preds = torch.sigmoid(model(X_test.to(device))).cpu().numpy()
            test_preds = (test_preds >= 0.5).astype(int)
            Y_test_np = Y_test.numpy().astype(int)

        # Compute F1 for each column simultaneously 
        for i in range(len(target_cols)):
            if len(np.unique(Y_test_np[:, i])) < 2:
                fold_f1_matrix[fold, i] = 0.5 # Baseline fallthrough
            else:
                fold_f1_matrix[fold, i] = f1_score(Y_test_np[:, i], test_preds[:, i], average="macro")

    # Mean across folds for each target column
    mean_target_f1s = np.mean(fold_f1_matrix, axis=0)
    return dict(zip(target_cols, mean_target_f1s))


def update_results_tsv(trial_name, overall_f1, dataset_f1s, description, file_path="results.tsv"):
    """Appends the results of this evaluation run into a tab-separated file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_data = {
        "trial_name": trial_name,
        "timestamp": timestamp,
        "f1": f"{overall_f1:.6f}",
        "health_f1": f"{dataset_f1s.get('health', 0.0):.6f}",
        "climate_f1": f"{dataset_f1s.get('climate', 0.0):.6f}",
        "fema_f1": f"{dataset_f1s.get('fema', 0.0):.6f}",
        "svi_f1": f"{dataset_f1s.get('svi', 0.0):.6f}",
        "description": description
    }
    df_new = pd.DataFrame([row_data])
    if os.path.exists(file_path):
        df_new.to_csv(file_path, mode='a', sep='\t', index=False, header=False)
    else:
        df_new.to_csv(file_path, mode='w', sep='\t', index=False, header=True)
    print(f"Logged run details to {os.path.abspath(file_path)}")


def evaluate_embeddings(parquet_path, trial_name="auto", desc=""):
    """Evaluate a GeoParquet embedding file on all downstream tasks.
    Returns (overall_f1, dataset_f1s_dict)."""
    datasets = ["health", "climate", "fema", "svi"]
    dataset_f1s = {}

    print(f"Loading embeddings from: {parquet_path}...")
    global_emb_gdf = gpd.read_parquet(parquet_path)

    for ds in datasets:
        data_gdf, target_cols = get_dataset_data(ds)
        if data_gdf is None or len(target_cols) == 0:
            continue

        print(f"Processing {ds.upper()} dataset with {len(target_cols)} targets...")
        try:
            X, aligned_gdf = get_spatial_embeddings(data_gdf, global_emb_gdf)
        except Exception as e:
            print(f"Failed spatial join mapping for {ds}: {e}. Skipping.")
            continue

        print(f"Training Multi-Task MLP over all targets for {ds}...")
        target_f1_dict = process_all_targets_mlp(X, aligned_gdf, target_cols)

        if target_f1_dict:
            dataset_f1s[ds] = np.mean(list(target_f1_dict.values()))
        else:
            dataset_f1s[ds] = 0.0

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)

    if dataset_f1s:
        overall_f1 = np.mean(list(dataset_f1s.values()))
        print(f"f1:   {overall_f1:.6f}")
        for ds, score in dataset_f1s.items():
            print(f"{ds}_f1: {score:.6f}")
        update_results_tsv(trial_name, overall_f1, dataset_f1s, desc)
    else:
        overall_f1 = None
        print("No evaluation datasets could be properly parsed.")
    print("=" * 40)
    return overall_f1, dataset_f1s


def main():
    parser = argparse.ArgumentParser(description="Evaluate Spatial Embeddings across multiple downstream Tasks.")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to Geoparquet embeddings file.")
    parser.add_argument("--trial_name", type=str, default="eval_run", help="Name of the trial run.")
    parser.add_argument("--desc", type=str, default="", help="Description regarding this setup.")
    args = parser.parse_args()
    evaluate_embeddings(args.parquet_path, args.trial_name, args.desc)


if __name__ == "__main__":
    main()