#!/usr/bin/env python3
"""
Shared utilities for Sentinel-2 tree-species classification pipeline.
"""
import os
import random
import rasterio
import numpy as np

ROOT_DIR = "s2"          # folder that holds s2/60m and s2/200m subfolders
TRAIN_LIST = "train_filenames.lst"
TEST_LIST  = "test_filenames.lst"

SRC_BAND_ORDER = [
    "B02", "B03", "B04", "B08", "B05", "B06", "B07",
    "B8A", "B11", "B12", "B01", "B09"
]
BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12"
]
REORDER_IDX = [SRC_BAND_ORDER.index(b) for b in BAND_ORDER]

def load_split_list(path: str) -> list[str]:
    """Return list of filenames (no path) from a .lst file."""
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def parse_species(fname: str) -> str:
    """Return genus_species or 'Cleared' from a patch filename."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    return "Cleared" if parts[0].lower() == "cleared" else f"{parts[0]}_{parts[1]}"

def get_paths_for_split(basenames: list[str], resolution: str) -> list[str]:
    """Full paths for the given basenames inside ROOT_DIR/<resolution>/."""
    dir_path = os.path.join(ROOT_DIR, resolution)
    return [os.path.join(dir_path, f) for f in basenames
            if os.path.isfile(os.path.join(dir_path, f))]

# --- Patch-level feature ----------------------------------------------------
def compute_patch_mean(path: str, reorder_idx: list[int] = REORDER_IDX) -> np.ndarray:
    """Read a patch, reorder bands, return mean reflectance per band (len 12)."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)          # shape (12, H, W)
    data = data[reorder_idx]                          # reorder to BAND_ORDER
    return data.reshape(data.shape[0], -1).mean(axis=1)
