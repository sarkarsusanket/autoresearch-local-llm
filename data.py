import numpy as np
from pathlib import Path

def load_data(data_dir="data"):
    data_dir = Path(data_dir)
    modalities = {}
    for fpath in sorted(data_dir.glob("*.npy")):
        name = fpath.stem
        modalities[name] = np.load(str(fpath))
    return modalities

def get_modality_dims(modalities):
    return {name: arr.shape[1] for name, arr in modalities.items()}
