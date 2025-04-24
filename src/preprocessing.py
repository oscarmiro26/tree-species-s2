#!/usr/bin/env python3
import os
import random
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Config
ROOT_DIR = "s2"  # contains s2/60m and s2/200m
TRAIN_LIST = "train_filenames.lst"
TEST_LIST = "test_filenames.lst"
OUTPUT_DIR = "results"
SAMPLE_SIZE = 200  # max patches per species for spectral comparison

SRC_BAND_ORDER = [
    "B02", "B03", "B04", "B08", "B05", "B06", "B07",
    "B8A", "B11", "B12", "B01", "B09"
]
BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12"
]
REORDER_IDX = [SRC_BAND_ORDER.index(b) for b in BAND_ORDER]


def load_split_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def parse_species(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if parts[0].lower() == "cleared":
        return "Cleared"
    return f"{parts[0]}_{parts[1]}"


def get_paths_for_species(basenames, resolution):
    species_map = {}
    res_dir = os.path.join(ROOT_DIR, resolution)
    for name in basenames:
        sp = parse_species(name)
        path = os.path.join(res_dir, name)
        if os.path.isfile(path):
            species_map.setdefault(sp, []).append(path)
        else:
            print(f"Warning: missing file {path}")
    return species_map


def plot_species_distribution(basenames, split):
    species = [parse_species(name) for name in basenames]
    counts = {}
    for sp in species:
        counts[sp] = counts.get(sp, 0) + 1
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels, freqs = zip(*items) if items else ([], [])
    plt.figure(figsize=(12, 6))
    plt.bar(labels, freqs)
    plt.xticks(rotation=90)
    plt.ylabel("Sample count")
    plt.title(f"Species distribution ({split})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"species_distribution_{split}.png"))
    plt.close()


def compute_spectra_stats(paths):
    """
    Load up to SAMPLE_SIZE patches, reorder bands, and compute mean and std per band.
    Returns (mean_spectrum, std_spectrum) arrays of shape (len(BAND_ORDER),).
    """
    selected = paths if len(paths) <= SAMPLE_SIZE else random.sample(paths, SAMPLE_SIZE)
    spectra = []
    for p in selected:
        with rasterio.open(p) as src:
            data = src.read().astype(np.float32)
        data = data[REORDER_IDX]  # reorder bands
        spectra.append(data.reshape(data.shape[0], -1).mean(axis=1))
    arr = np.stack(spectra)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_spectral_comparison(species_map, split, resolution):
    plt.figure(figsize=(10, 6))
    for sp, paths in species_map.items():
        if not paths:
            continue
        mean_spec, _ = compute_spectra_stats(paths)
        plt.plot(range(len(BAND_ORDER)), mean_spec, marker='o', label=sp)
    plt.xticks(range(len(BAND_ORDER)), BAND_ORDER, rotation=45)
    plt.xlabel("Band")
    plt.ylabel("Mean reflectance")
    plt.title(f"Spectral comparison ({split}, {resolution})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    fname = f"spectral_comparison_{split}_{resolution}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()


def plot_spectral_errorbars(species_map, split, resolution):
    plt.figure(figsize=(10, 6))
    for sp, paths in species_map.items():
        if not paths:
            continue
        mean_spec, std_spec = compute_spectra_stats(paths)
        plt.errorbar(
            range(len(BAND_ORDER)), mean_spec, yerr=std_spec,
            fmt='-o', capsize=3, label=sp
        )
    plt.xticks(range(len(BAND_ORDER)), BAND_ORDER, rotation=45)
    plt.xlabel("Band")
    plt.ylabel("Mean reflectance ± STD")
    plt.title(f"Spectral comparison with error bars ({split}, {resolution})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    fname = f"spectral_comparison_errorbars_{split}_{resolution}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train = load_split_list(TRAIN_LIST)
    test = load_split_list(TEST_LIST)
    for split, basenames in [('train', train), ('test', test)]:
        # species distribution
        plot_species_distribution(basenames, split)
        # spectral comparison and error bars
        for res in ['60m', '200m']:
            species_map = get_paths_for_species(basenames, res)
            plot_spectral_comparison(species_map, split, res)
            plot_spectral_errorbars(species_map, split, res)

if __name__ == '__main__':
    main()
