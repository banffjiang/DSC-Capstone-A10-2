#!/usr/bin/env python3
"""
AoA Analysis Script for BabyLM Evaluation Pipeline
====================================================
Following Chang & Bergen (2022) methodology:
1. Compute mean surprisal per word per checkpoint
2. Fit sigmoid curves to learning trajectories
3. Derive model AoA (word count where surprisal reaches 50% threshold)
4. Correlate with human CDI AoA data (Spearman correlation)
5. Generate diagnostic plots
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. Data Loading
# ============================================================

def load_surprisal(path: str) -> dict:
    """Load surprisal.json and return structured data."""
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    metadata = data.get("metadata", {})

    # Build step -> word_count mapping
    step_to_wc = {}
    for r in results:
        step_to_wc[r["step"]] = r["word_count"]

    # Sort steps by word_count
    sorted_steps = sorted(step_to_wc.keys(), key=lambda s: step_to_wc[s])

    # Build word -> {step -> [surprisal values]}
    word_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        word_data[r["target_word"]][r["step"]].append(r["surprisal"])

    return {
        "results": results,
        "metadata": metadata,
        "step_to_wc": step_to_wc,
        "sorted_steps": sorted_steps,
        "word_data": dict(word_data),
    }


def compute_mean_surprisals(data: dict) -> dict:
    """Compute mean surprisal per word per step."""
    word_means = {}
    for word, step_dict in data["word_data"].items():
        word_means[word] = {}
        for step in data["sorted_steps"]:
            vals = step_dict.get(step, [])
            if vals:
                # Filter NaN values
                valid = [v for v in vals if np.isfinite(v)]
                word_means[word][step] = np.mean(valid) if valid else np.nan
            else:
                word_means[word][step] = np.nan
    return word_means


# ============================================================
# 2. Sigmoid Fitting & Model AoA
# ============================================================

def sigmoid(x, L, k, x0, b):
    """4-parameter sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def fit_sigmoid_to_word(word_counts, mean_surprisals):
    """Fit a sigmoid curve to a word's surprisal trajectory.
    """
    # Filter valid data points
    valid = [(wc, s) for wc, s in zip(word_counts, mean_surprisals) if np.isfinite(s)]
    if len(valid) < 4:
        return None, False

    x = np.array([np.log10(wc) for wc, _ in valid])
    y = np.array([s for _, s in valid])

    # Initial guesses
    L0 = y[0] - y[-1]    # amplitude (surprisal should decrease)
    b0 = y[-1]           # baseline (final surprisal)
    x0_0 = np.mean(x)    # midpoint
    k0 = -1.0            # negative slope (decreasing)

    try:
        popt, _ = curve_fit(
            sigmoid, x, y,
            p0=[L0, k0, x0_0, b0],
            maxfev=10000,
            bounds=(
                [-np.inf, -np.inf, x.min() - 2, -np.inf],
                [np.inf, np.inf, x.max() + 2, np.inf]
            ),
        )
        return popt, True
    except (RuntimeError, ValueError):
        return None, False


def compute_model_aoa(params, word_counts, mean_surprisals):
    """Compute model AoA as log10(word_count) where surprisal reaches
    50% between initial (chance) surprisal and minimum surprisal.
    """
    valid_s = [s for s in mean_surprisals if np.isfinite(s)]
    if not valid_s or params is None:
        return np.nan

    U = max(valid_s)  # upper bound (chance-level surprisal)
    L = min(valid_s)  # lower bound (best surprisal)
    threshold = (U + L) / 2.0

    # Find where sigmoid crosses threshold
    L_sig, k, x0, b = params
    x_range = np.linspace(
        np.log10(min(word_counts)) - 0.5,
        np.log10(max(word_counts)) + 0.5,
        10000,
    )
    y_fitted = sigmoid(x_range, *params)

    # Find crossing point
    crossings = np.where(np.diff(np.sign(y_fitted - threshold)))[0]
    if len(crossings) > 0:
        return x_range[crossings[0]]
    else:
        # If no crossing, use midpoint parameter
        return x0


def compute_all_model_aoa(data: dict, word_means: dict) -> dict:
    """Compute model AoA for all words."""
    step_to_wc = data["step_to_wc"]
    sorted_steps = data["sorted_steps"]
    word_counts = [step_to_wc[s] for s in sorted_steps]

    results = {}
    for word in word_means:
        surprisals = [word_means[word].get(s, np.nan) for s in sorted_steps]
        params, success = fit_sigmoid_to_word(word_counts, surprisals)

        if success:
            aoa = compute_model_aoa(params, word_counts, surprisals)
        else:
            aoa = np.nan

        valid_s = [s for s in surprisals if np.isfinite(s)]
        results[word] = {
            "model_aoa_log10": aoa,
            "sigmoid_params": list(params) if params is not None else None,
            "sigmoid_fit_success": success,
            "mean_surprisal_final": surprisals[-1] if surprisals else np.nan,
            "surprisal_range": (max(valid_s) - min(valid_s)) if valid_s else 0,
            "mean_surprisals": surprisals,
        }

    return results


# ============================================================
# 3. Human AoA (Wordbank CDI)
# ============================================================

def load_human_aoa_csv(path: str) -> dict:
    """Load human AoA data from a CSV file.
    """
    import csv
    human_aoa = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", row.get("item_definition", "")).strip().lower()
            aoa = row.get("aoa", row.get("age_of_acquisition", ""))
            if word and aoa:
                try:
                    human_aoa[word] = float(aoa)
                except ValueError:
                    pass
    return human_aoa

# ============================================================
# 4. Correlation Analysis
# ============================================================

def correlate_aoa(model_aoa: dict, human_aoa: dict) -> dict:
    """Compute Spearman correlation between model and human AoA."""
    matched_words = []
    model_vals = []
    human_vals = []

    for word in model_aoa:
        word_lower = word.lower()
        if word_lower in human_aoa and np.isfinite(model_aoa[word]["model_aoa_log10"]):
            matched_words.append(word)
            model_vals.append(model_aoa[word]["model_aoa_log10"])
            human_vals.append(human_aoa[word_lower])

    if len(matched_words) < 5:
        print(f"WARNING: Only {len(matched_words)} matched words. Need at least 5.")
        return {"correlation": np.nan, "p_value": np.nan, "n_matched": len(matched_words)}

    rho, p_val = spearmanr(model_vals, human_vals)

    return {
        "spearman_rho": rho,
        "p_value": p_val,
        "n_matched": len(matched_words),
        "matched_words": matched_words,
        "model_aoa_values": model_vals,
        "human_aoa_values": human_vals,
    }


# ============================================================
# 5. Plotting
# ============================================================

def plot_learning_curves(data, word_means, model_aoa_results, output_dir, n_words=12):
    """Plot learning curves for a sample of words."""
    step_to_wc = data["step_to_wc"]
    sorted_steps = data["sorted_steps"]
    word_counts = [step_to_wc[s] for s in sorted_steps]
    log_wc = [np.log10(wc) for wc in word_counts]

    # Select words with successful sigmoid fits and diverse AoA
    fitted_words = [
        w for w, r in model_aoa_results.items()
        if r["sigmoid_fit_success"] and np.isfinite(r["model_aoa_log10"])
    ]
    fitted_words.sort(key=lambda w: model_aoa_results[w]["model_aoa_log10"])

    if len(fitted_words) == 0:
        print("No words with successful sigmoid fits to plot.")
        return

    # Sample evenly across AoA range
    indices = np.linspace(0, len(fitted_words) - 1, min(n_words, len(fitted_words)), dtype=int)
    sample_words = [fitted_words[i] for i in indices]

    ncols = 4
    nrows = (len(sample_words) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), squeeze=False)

    x_fine = np.linspace(min(log_wc) - 0.2, max(log_wc) + 0.2, 200)

    for idx, word in enumerate(sample_words):
        row, col = idx // ncols, idx % ncols
        ax = axes[row][col]

        surprisals = [word_means[word].get(s, np.nan) for s in sorted_steps]
        valid_s = [s for s in surprisals if np.isfinite(s)]

        # Plot raw data
        ax.plot(log_wc, surprisals, "ko", markersize=5, label="Data")

        # Plot fitted sigmoid
        params = model_aoa_results[word]["sigmoid_params"]
        if params is not None:
            y_fit = sigmoid(x_fine, *params)
            ax.plot(x_fine, y_fit, "b-", linewidth=1.5, label="Sigmoid fit")

        # Plot AoA threshold
        if valid_s:
            threshold = (max(valid_s) + min(valid_s)) / 2.0
            ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.5, label="AoA threshold")

        aoa = model_aoa_results[word]["model_aoa_log10"]
        if np.isfinite(aoa):
            ax.axvline(x=aoa, color="g", linestyle="--", alpha=0.5, label=f"AoA={aoa:.2f}")

        ax.set_title(f'"{word}"', fontsize=11)
        ax.set_xlabel("log₁₀(word count)")
        ax.set_ylabel("Surprisal")
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide unused subplots
    for idx in range(len(sample_words), nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row][col].set_visible(False)

    fig.suptitle("Word Learning Curves with Sigmoid Fits", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'learning_curves.png'}")


def plot_aoa_distribution(model_aoa_results, output_dir):
    """Plot distribution of model AoA values."""
    aoa_vals = [
        r["model_aoa_log10"]
        for r in model_aoa_results.values()
        if np.isfinite(r["model_aoa_log10"])
    ]

    if not aoa_vals:
        print("No valid AoA values to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(aoa_vals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Model AoA (log₁₀ word count)", fontsize=12)
    ax.set_ylabel("Number of Words", fontsize=12)
    ax.set_title("Distribution of Model Age of Acquisition", fontsize=14)
    ax.axvline(np.median(aoa_vals), color="red", linestyle="--", label=f"Median = {np.median(aoa_vals):.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "aoa_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'aoa_distribution.png'}")


def plot_correlation(corr_results, output_dir):
    """Plot model AoA vs human AoA scatter plot."""
    if corr_results["n_matched"] < 5:
        print("Not enough matched words to plot correlation.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        corr_results["human_aoa_values"],
        corr_results["model_aoa_values"],
        alpha=0.5, s=20, color="steelblue",
    )

    # Add word labels for a subset
    n_label = min(20, len(corr_results["matched_words"]))
    indices = np.linspace(0, len(corr_results["matched_words"]) - 1, n_label, dtype=int)
    for i in indices:
        ax.annotate(
            corr_results["matched_words"][i],
            (corr_results["human_aoa_values"][i], corr_results["model_aoa_values"][i]),
            fontsize=7, alpha=0.7,
        )

    rho = corr_results["spearman_rho"]
    p = corr_results["p_value"]
    n = corr_results["n_matched"]
    ax.set_xlabel("Human AoA (months)", fontsize=12)
    ax.set_ylabel("Model AoA (log₁₀ word count)", fontsize=12)
    ax.set_title(f"Model vs Human AoA (ρ={rho:.3f}, p={p:.2e}, n={n})", fontsize=13)

    # Trend line
    z = np.polyfit(corr_results["human_aoa_values"], corr_results["model_aoa_values"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(corr_results["human_aoa_values"]), max(corr_results["human_aoa_values"]), 100)
    ax.plot(x_line, p_line(x_line), "r--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "aoa_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'aoa_correlation.png'}")


def plot_surprisal_heatmap(data, word_means, model_aoa_results, output_dir, n_words=50):
    """Plot heatmap of surprisal across words and checkpoints."""
    sorted_steps = data["sorted_steps"]
    step_to_wc = data["step_to_wc"]

    # Select words sorted by AoA
    fitted_words = [
        w for w, r in model_aoa_results.items()
        if r["sigmoid_fit_success"] and np.isfinite(r["model_aoa_log10"])
    ]
    fitted_words.sort(key=lambda w: model_aoa_results[w]["model_aoa_log10"])

    if len(fitted_words) == 0:
        return

    sample = fitted_words[:n_words]

    matrix = np.zeros((len(sample), len(sorted_steps)))
    for i, word in enumerate(sample):
        for j, step in enumerate(sorted_steps):
            matrix[i, j] = word_means[word].get(step, np.nan)

    fig, ax = plt.subplots(figsize=(10, max(6, len(sample) * 0.25)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")

    step_labels = [f"{step_to_wc[s] / 1e6:.0f}M" for s in sorted_steps]
    ax.set_xticks(range(len(sorted_steps)))
    ax.set_xticklabels(step_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(sample)))
    ax.set_yticklabels(sample, fontsize=7)
    ax.set_xlabel("Training Tokens")
    ax.set_ylabel("Words (sorted by model AoA)")
    ax.set_title("Surprisal Heatmap (sorted by model AoA)")
    plt.colorbar(im, ax=ax, label="Surprisal")
    plt.tight_layout()
    plt.savefig(output_dir / "surprisal_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'surprisal_heatmap.png'}")


# ============================================================
# 6. Summary Statistics
# ============================================================

def print_summary(model_aoa_results, corr_results=None):
    """Print summary statistics."""
    total = len(model_aoa_results)
    fitted = sum(1 for r in model_aoa_results.values() if r["sigmoid_fit_success"])
    valid_aoa = sum(
        1 for r in model_aoa_results.values()
        if np.isfinite(r["model_aoa_log10"])
    )

    print("\n" + "=" * 60)
    print("AoA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total words evaluated:       {total}")
    print(f"Successful sigmoid fits:     {fitted} ({100*fitted/total:.1f}%)")
    print(f"Valid model AoA computed:    {valid_aoa} ({100*valid_aoa/total:.1f}%)")

    aoa_vals = [
        r["model_aoa_log10"]
        for r in model_aoa_results.values()
        if np.isfinite(r["model_aoa_log10"])
    ]
    if aoa_vals:
        print(f"\nModel AoA (log10 word count):")
        print(f"  Mean:   {np.mean(aoa_vals):.3f}")
        print(f"  Median: {np.median(aoa_vals):.3f}")
        print(f"  Std:    {np.std(aoa_vals):.3f}")
        print(f"  Min:    {np.min(aoa_vals):.3f}")
        print(f"  Max:    {np.max(aoa_vals):.3f}")

    # Top 10 earliest and latest acquired words
    sorted_words = sorted(
        [(w, r["model_aoa_log10"]) for w, r in model_aoa_results.items()
         if np.isfinite(r["model_aoa_log10"])],
        key=lambda x: x[1],
    )
    if sorted_words:
        print(f"\nEarliest acquired words (model):")
        for w, a in sorted_words[:10]:
            print(f"  {w:20s}  AoA = {a:.3f}")
        print(f"\nLatest acquired words (model):")
        for w, a in sorted_words[-10:]:
            print(f"  {w:20s}  AoA = {a:.3f}")

    if corr_results and corr_results.get("n_matched", 0) >= 5:
        print(f"\nCorrelation with Human CDI AoA:")
        print(f"  Spearman ρ:  {corr_results['spearman_rho']:.4f}")
        print(f"  p-value:     {corr_results['p_value']:.2e}")
        print(f"  N matched:   {corr_results['n_matched']}")

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AoA Analysis for BabyLM")
    parser.add_argument(
        "--surprisal", type=str, required=True,
        help="Path to surprisal.json from evaluation pipeline",
    )
    parser.add_argument(
        "--output", type=str, default="aoa_results",
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--human_aoa", type=str, default=None,
        help="Path to human AoA CSV file (columns: word, aoa)",
    )
    parser.add_argument(
        "--download_wordbank", action="store_true",
        help="Print instructions for downloading Wordbank data",
    )
    parser.add_argument(
        "--n_plot_words", type=int, default=12,
        help="Number of words to show in learning curves plot",
    )
    args = parser.parse_args()

    if args.download_wordbank:
        download_wordbank_aoa()
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process data
    print("Loading surprisal data...")
    data = load_surprisal(args.surprisal)
    print(f"  {len(data['word_data'])} words, {len(data['sorted_steps'])} checkpoints")

    print("Computing mean surprisals...")
    word_means = compute_mean_surprisals(data)

    print("Fitting sigmoid curves and computing model AoA...")
    model_aoa_results = compute_all_model_aoa(data, word_means)

    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(data, word_means, model_aoa_results, output_dir, args.n_plot_words)
    plot_aoa_distribution(model_aoa_results, output_dir)
    plot_surprisal_heatmap(data, word_means, model_aoa_results, output_dir)

    # Human AoA correlation
    corr_results = None
    if args.human_aoa:
        print(f"\nLoading human AoA from {args.human_aoa}...")
        human_aoa = load_human_aoa_csv(args.human_aoa)
        print(f"  Loaded {len(human_aoa)} human AoA entries")

        corr_results = correlate_aoa(model_aoa_results, human_aoa)
        plot_correlation(corr_results, output_dir)

    # Save results JSON
    save_data = {
        "metadata": data["metadata"],
        "step_to_word_count": data["step_to_wc"],
        "model_aoa": {
            word: {
                "model_aoa_log10": float(r["model_aoa_log10"]) if np.isfinite(r["model_aoa_log10"]) else None,
                "sigmoid_fit_success": r["sigmoid_fit_success"],
                "mean_surprisal_final": float(r["mean_surprisal_final"]) if np.isfinite(r["mean_surprisal_final"]) else None,
                "surprisal_range": float(r["surprisal_range"]),
            }
            for word, r in model_aoa_results.items()
        },
    }
    if corr_results and corr_results.get("n_matched", 0) >= 5:
        save_data["correlation"] = {
            "spearman_rho": float(corr_results["spearman_rho"]),
            "p_value": float(corr_results["p_value"]),
            "n_matched": corr_results["n_matched"],
        }

    results_path = output_dir / "aoa_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Print summary
    print_summary(model_aoa_results, corr_results)


if __name__ == "__main__":
    main()