#!/usr/bin/env python3
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from utils import (
    load_split_list,
    parse_species,
    get_paths_for_split,
    compute_patch_mean,
    BAND_ORDER,
    REORDER_IDX
)

# Configuration
TRAIN_LIST = "train_filenames.lst"
RESOLUTIONS = ["60m", "200m"]
OUTPUT_DIR = "results/eda"
MAX_SAMPLES_PER_SPECIES = 200  # cap to avoid overload
RANDOM_STATE = 42


def gather_data(split_files, resolution):
    """
    Sample patches per species and compute mean spectra.
    Returns X (n_samples x n_bands), y (n_samples labels).
    """
    species_map = {}
    paths = get_paths_for_split(split_files, resolution)   # drop the extra "s2"
    for p in paths:
        sp = parse_species(os.path.basename(p))
        species_map.setdefault(sp, []).append(p)
    X, y = [], []
    for sp, ps in species_map.items():
        selected = ps if len(ps) <= MAX_SAMPLES_PER_SPECIES else random.Random(RANDOM_STATE).sample(ps, MAX_SAMPLES_PER_SPECIES)
        for p in selected:
            vec = compute_patch_mean(p)
            X.append(vec)
            y.append(sp)
    return np.stack(X), np.array(y)


def plot_boxplots(X, y, resolution):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    species = np.unique(y)
    n_bands = X.shape[1]
    data_by_band = []
    for i in range(n_bands):
        vals = [X[y == sp, i] for sp in species]
        data_by_band.append(vals)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes[:n_bands]):
        ax.boxplot(data_by_band[i], tick_labels=species, showfliers=False)
        ax.set_title(BAND_ORDER[i])
        ax.tick_params(axis='x', rotation=90)
    fig.suptitle(f"Boxplots of mean reflectance by species ({resolution})")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplots_{resolution}.png"), dpi=300)
    plt.close()


def plot_correlation_heatmap(X, resolution):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    corr = np.corrcoef(X.T)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(BAND_ORDER)), BAND_ORDER, rotation=45)
    plt.yticks(range(len(BAND_ORDER)), BAND_ORDER)
    plt.title(f"Band correlation heatmap ({resolution})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"corr_heatmap_{resolution}.png"), dpi=300)
    plt.close()


def plot_pca_scatter(X, y, resolution):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_scaled = StandardScaler().fit_transform(X)
    pc = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    species = np.unique(y)
    plt.figure(figsize=(10, 8))
    for sp in species:
        mask = y == sp
        plt.scatter(pc[mask, 0], pc[mask, 1], label=sp, s=20, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA scatter ({resolution})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"pca_scatter_{resolution}.png"), dpi=300)
    plt.close()


def numeric_eda(X, y):
    """
    Compute and print ANOVA F-statistics, mutual information, and silhouette score.
    """
    species = np.unique(y)
    print("\n=== ANOVA F-statistics per band ===")
    f_stats = []
    for i, band in enumerate(BAND_ORDER):
        groups = [X[y == sp, i] for sp in species if np.sum(y == sp) > 1]
        if len(groups) > 1:
            fval, pval = f_oneway(*groups)
        else:
            fval, pval = np.nan, np.nan
        f_stats.append((band, fval, pval))
    for band, fval, pval in f_stats:
        print(f"{band}: F={fval:.3f}, p={pval:.3e}")

    print("\n=== Mutual Information per band ===")
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_STATE)
    for band, val in zip(BAND_ORDER, mi):
        print(f"{band}: MI={val:.3f}")

    print("\n=== Silhouette score on 2D PCA ===")
    try:
        sil = silhouette_score(PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(StandardScaler().fit_transform(X)), y)
        print(f"Silhouette score: {sil:.3f}")
    except Exception as e:
        print(f"Silhouette score error: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_files = load_split_list(TRAIN_LIST)
    for res in RESOLUTIONS:
        X, y = gather_data(train_files, res)
        plot_boxplots(X, y, res)
        plot_correlation_heatmap(X, res)
        plot_pca_scatter(X, y, res)
        numeric_eda(X, y)

if __name__ == '__main__':
    main()
