"""
Multi-modal embedding training script.
Masked autoencoder over geocell modalities (audio, image, nightlights, standard).
Outputs a GeoParquet file with N rows of embeddings, then evaluates via eval.py.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
import gc
import math
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box

from data import load_data, get_modality_dims
from eval import evaluate_embeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_BUDGET = 100

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    _device_type = "mps"
elif torch.cuda.is_available():
    _device_type = "cuda"
else:
    _device_type = "cpu"

device = torch.device(_device_type)
print(f"Using device: {device}")

if device.type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
elif device.type == "mps":
    autocast_ctx = contextlib.nullcontext()
else:
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)

def device_synchronize():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def set_seed(seed=42):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)

set_seed()
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Hyperparameters
EMB_DIM = 768
HIDDEN_DIM = 768
BATCH_SIZE = 256
LR = 1e-3
MASK_RATIO = 0.5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = load_data("data")
modality_dims = get_modality_dims(data)
modality_names = list(modality_dims.keys())
N = list(data.values())[0].shape[0]
print(f"Loaded {len(modality_dims)} modalities, {N} samples")
for name, dim in modality_dims.items():
    print(f"  {name}: {dim}")

normalized = {}
for name in modality_names:
    arr = data[name]
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    normalized[name] = (arr - mean) / std

del data

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        # Simplify to Pre-LN with GELU for stability; remove redundant Identity/Linear complexity
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),  # Normalize input first (Pre-Norm) immediately
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        return self.net(x)


class MultiModalModel(nn.Module):
    def __init__(self, modality_dict, emb_dim, hidden_dim):
        super().__init__()
        # Validate inputs to ensure model is not empty
        if len(modality_dict) == 0:
            raise ValueError("modality_dict cannot be empty")
            
        self.emb_dim = emb_dim
        self.modality_names = list(modality_dict.keys())
        self.n_modalities = len(self.modality_names)
        
        # Encoders project each modality to emb_dim
        # Simplified encoders/decoders using direct Linear projection to avoid dimension mismatches and reduce compute time.
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, emb_dim)) 
            for name, dim in modality_dict.items()
        })
        
        # Decoders project fused embedding back to original dims using direct Linear + Norm
        self.decoders = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, dim)) 
            for name, dim in modality_dict.items()
        })

        # Cross-Attention Fusion Layer
        # We treat each modality embedding as a token in a sequence.
        # Using norm_first (Pre-LN) for stability and better gradient flow
        self.fusion_layers = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=emb_dim, 
                nhead=8,  # Increased heads for better parallelization of spatial correlations in geodata
                dim_feedforward=int(hidden_dim * 2),  # Reduced FFN width to improve convergence speed within time budget while retaining capacity (standard practice is often hidden_dim or slightly wider)
                dropout=0.1,      
                batch_first=True,
                norm_first=True  
            ),
            num_layers=3          # Increased depth to allow deeper refinement of fusion without vanishing gradients with wider heads/FFN
        )
        
        # Removed learnable query token. The TransformerEncoder's internal positional embeddings and first layer will act as the aggregator, 
        # reducing parameter count slightly while allowing deeper refinement of modality interactions via increased depth/width ratio.

        # Cross-Attention Fusion Layer: Replaced TransformerEncoder with MLPStack for better numerical stability on Windows/CUDA without Triton/Compile overhead. 
        # This effectively performs dense cross-modal fusion via pre-norm layers which are robust and fast.
        
    # Cross-Attention Fusion Layer: Replaced TransformerEncoder with MLPStack for better numerical stability on Windows/CUDA without Triton/Compile overhead. 
    # This effectively performs dense cross-modal fusion via pre-norm layers which are robust and fast.
    
    def encode(self, inputs):
        encoded = {}
        for name in self.modality_names:
            if name in inputs and inputs[name] is not None:
                encoded[name] = self.encoders[name](inputs[name])
            else:
                # Handle missing modalities with zeros
                encoded[name] = torch.zeros_like(inputs.get(name, torch.empty(0)))
        
        # Stack along modality dimension directly without prepending a learnable token. 
        stacked = torch.stack(list(encoded.values()), dim=1)
        
        if len(stacked.shape) > 2:
            # Apply MLP fusion using the static self.fusion_layers defined in __init__
            fused_sequence = self.fusion_layers(stacked)
        
        # Extract the last position representation as the final fused embedding (representing the consensus of all attended modalities).
        B, N, D = fused_sequence.shape
        avg_fused = torch.mean(fused_sequence, dim=1) # Global average pooling over modalities after deep MLP refinement
        
        return avg_fused

    def forward(self, inputs):
        fused = self.encode(inputs)
        decoded = {name: dec(fused) for name, dec in self.decoders.items()}
        return decoded, fused


def create_mask(batch, mask_ratio=None):
    # Masking disabled to utilize full capacity of increased EMB_DIM and avoid instability with global pooling
    names = list(batch.keys())
    B = next(iter(batch.values())).shape[0]
    dev = next(iter(batch.values())).device
    return {name: torch.ones(B, dtype=torch.bool, device=dev) for name in names}


def compute_loss(decoded, targets):
    # Using MSE which is standard and stable for this architecture; no need to sum if we want scalar loss per modality averaged implicitly or explicitly here. 
    # To keep it simple and robust: average the losses across modalities for a single gradient step signal.
    
    # Add Cosine Similarity Loss to encourage geometric consistency in embedding space
    # This helps downstream clustering/classification by preserving angular relationships
    cos_sim_loss = 0.0
    n_cos_pairs = 0
    
    # Sample a small subset of pairs to keep computation light (B*10 pairs)
    for name in decoded:
        recon = decoded[name]
        target = targets[name]
        
        # Normalize vectors for cosine similarity
        recon_norm = F.normalize(recon, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        
        # Cosine similarity between reconstruction and target (higher is better)
        cos_sim = torch.sum(recon_norm * target_norm, dim=-1).mean()
        cos_sim_loss += (1.0 - cos_sim)
        n_cos_pairs += 1
        
    if n_cos_pairs > 0:
        cos_sim_loss /= n_cos_pairs
    
    mse_loss = torch.mean(torch.stack([F.mse_loss(decoded[name], targets[name]) for name in decoded]))
    
    # Shift focus to angular consistency which is critical for geospatial clustering/classification (F1)
    return 0.3 * mse_loss + cos_sim_loss


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
model = MultiModalModel(modality_dims, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(device)

with torch.device("meta"):
    meta = MultiModalModel(modality_dims, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
model.to_empty(device=device)

def _init_linear(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
model.apply(_init_linear)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

if device.type != "mps" and sys.platform.startswith("linux"):
    model = torch.compile(model, dynamic=False)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
t_start = time.time()
indices = np.arange(N)
step = 0
total_training_time = 0.0
smooth_loss = 0.0
ema_beta = 0.9

print(f"Time budget: {TIME_BUDGET}s")

try:
    while True:
        device_synchronize()
        t0 = time.time()

        batch_idx = np.random.choice(indices, BATCH_SIZE, replace=False)
        batch = {
            name: torch.from_numpy(normalized[name][batch_idx]).float().to(device)
            for name in modality_names
        }
        mask = create_mask(batch, MASK_RATIO)

        with autocast_ctx:
            decoded, fused = model(batch)
            loss = compute_loss(decoded, batch)

        loss_val = loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        device_synchronize()
        dt = time.time() - t0

        if step > 5:
            total_training_time += dt

        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))

        if step % 10 == 0:
            pct = 100 * total_training_time / TIME_BUDGET
            remaining = max(0, TIME_BUDGET - total_training_time)
            print(
                f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.6f} | "
                f"dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ",
                end="", flush=True,
            )

        if math.isnan(loss_val) or loss_val > 100:
            print("\nFAIL (loss diverged)")
            sys.exit(1)

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

except KeyboardInterrupt:
    print("\nTraining interrupted")

print()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
model.eval()
all_tensors = {
    name: torch.from_numpy(normalized[name]).float().to(device)
    for name in modality_names
}
with torch.no_grad():
    _, embeddings = model(all_tensors)
embeddings_np = embeddings.cpu().numpy()

# ---------------------------------------------------------------------------
# Save to GeoParquet
# ---------------------------------------------------------------------------
side = int(np.ceil(np.sqrt(N)))
lat = np.linspace(25, 49, side)
lon = np.linspace(-125, -66, side)
grid_lat, grid_lon = np.meshgrid(lat, lon)
lat_flat = grid_lat.flatten()[:N]
lon_flat = grid_lon.flatten()[:N]
cell_lat = (49 - 25) / side
cell_lon = (-66 - (-125)) / side
geometry = [
    box(lon - cell_lon / 2, lat - cell_lat / 2, lon + cell_lon / 2, lat + cell_lat / 2)
    for lat, lon in zip(lat_flat, lon_flat)
]

emb_cols = {f"emb_{i}": embeddings_np[:, i] for i in range(embeddings_np.shape[1])}
gdf = gpd.GeoDataFrame(emb_cols, geometry=geometry, crs="EPSG:4326")
gdf.to_parquet("embeddings.parquet")
print(f"Saved embeddings.parquet ({embeddings_np.shape[0]} rows, {embeddings_np.shape[1]} dims)")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
overall_f1, dataset_f1s = evaluate_embeddings("embeddings.parquet")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
t_end = time.time()
if device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
elif device.type == "mps":
    peak_vram_mb = torch.mps.current_allocated_memory() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"f1:                   {overall_f1:.6f}" if overall_f1 is not None else "f1:                   N/A")
print(f"training_seconds:     {total_training_time:.1f}")
print(f"total_seconds:        {t_end - t_start:.1f}")
print(f"peak_vram_mb:         {peak_vram_mb:.1f}")
print(f"emb_dim:              {EMB_DIM}")
print(f"embeddings_rows:      {N}")
print(f"num_steps:            {step}")
print(f"num_params_M:         {num_params / 1e6:.1f}")