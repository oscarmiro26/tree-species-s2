import numpy as np, rasterio, torch
from torch.utils.data import Dataset
from pathlib import Path

BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]

class S2PatchDataset(Dataset):
    def __init__(self, lst_path, root, scale="60m", keep=BANDS[:10], as_tensor=True):
        self.root   = Path(root)
        self.paths  = [p.strip() for p in open(lst_path)]
        self.scale  = scale   # "60m" or "200m"
        self.keepix = [BANDS.index(b) for b in keep]
        self.as_tensor = as_tensor

    def __len__(self): return len(self.paths)

    def _label(self, fname): return int(fname.split("_")[0])   # 0‑based ID

    def __getitem__(self, idx):
        rel = self.paths[idx]
        path = self.root/self.scale/rel
        with rasterio.open(path) as src:
            arr = src.read(out_dtype="float32") / 1e4    # →0‑1
        arr = arr[self.keepix]
        if self.as_tensor: arr = torch.from_numpy(arr)
        label = self._label(rel)
        return arr, label
