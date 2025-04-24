#!/usr/bin/env python3
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    ROOT_DIR, TRAIN_LIST, TEST_LIST, BAND_ORDER, REORDER_IDX,
    load_split_list, parse_species, get_paths_for_split, compute_patch_mean
)

OUTPUT_DIR = "results"          # where plots are saved
SAMPLE_SIZE = 300                # max patches per species for spectral graphs


def build_species_map(basenames, resolution):
    paths = get_paths_for_split(basenames, resolution)
    species_map = {}
    for p in paths:
        sp = parse_species(os.path.basename(p))
        species_map.setdefault(sp, []).append(p)
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
    sel = paths if len(paths) <= SAMPLE_SIZE else random.sample(paths, SAMPLE_SIZE)
    arr = [compute_patch_mean(p) for p in sel]
    arr = np.stack(arr)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_spectral_comparison(species_map, split, resolution, errorbars=False):
    plt.figure(figsize=(10, 6))
    for sp, paths in species_map.items():
        if not paths:
            continue
        mean_spec, std_spec = compute_spectra_stats(paths)
        if errorbars:
            plt.errorbar(range(len(BAND_ORDER)), mean_spec, yerr=std_spec,
                         fmt='-o', capsize=3, label=sp)
        else:
            plt.plot(range(len(BAND_ORDER)), mean_spec, marker='o', label=sp)
    plt.xticks(range(len(BAND_ORDER)), BAND_ORDER, rotation=45)
    plt.xlabel("Band")
    plt.ylabel("Mean reflectance" + (" ± STD" if errorbars else ""))
    suffix = "errorbars" if errorbars else ""
    plt.title(f"Spectral comparison {suffix} ({split}, {resolution})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    tag = f"spectral_comparison_{suffix}_{split}_{resolution}".strip("_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}.png"), dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_names = load_split_list(TRAIN_LIST)
    test_names  = load_split_list(TEST_LIST)

    for split, names in [("train", train_names), ("test", test_names)]:
        plot_species_distribution(names, split)
        for res in ["60m", "200m"]:
            sp_map = build_species_map(names, res)
            plot_spectral_comparison(sp_map, split, res, errorbars=False)
            plot_spectral_comparison(sp_map, split, res, errorbars=True)

if __name__ == "__main__":
    main()
