#!/usr/bin/env python3
"""
Entropy Metrics Visualization Example

Demonstrates entropy analysis with multiple signal types.
Generates comparison plots and summary statistics.

Usage:
    python visualize_entropy.py

Output:
    entropy_visualization.png - Visual comparison of entropy metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def generate_signal(signal_type, duration=1.0, fs=1000):
    """Generate test signal of specified type.

    Args:
        signal_type: One of 'sine', 'noise', 'chirp', 'am_fm'
        duration: Signal duration in seconds (default: 1.0)
        fs: Sampling rate in Hz (default: 1000)

    Returns:
        numpy array of signal samples
    """
    t = np.linspace(0, duration, int(fs * duration))

    if signal_type == "sine":
        # Pure 5 Hz sinusoid
        return np.sin(2 * np.pi * 5 * t)

    elif signal_type == "noise":
        # White noise with low amplitude
        return np.random.randn(len(t)) * 0.1

    elif signal_type == "chirp":
        # Linear frequency sweep from 1 Hz to 20 Hz
        freq = np.linspace(1, 20, len(t))
        phase = 2 * np.pi * np.cumsum(freq) / len(t)
        return np.sin(phase)

    elif signal_type == "am_fm":
        # Amplitude modulated + frequency modulated
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        fm = np.sin(2 * np.pi * (5 + 5 * np.sin(2 * np.pi * 0.5 * t)) * t)
        return am * fm

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def spectral_entropy(signal):
    """Compute normalized spectral entropy.

    Measures frequency domain disorder (0=narrowband, 1=broadband).
    """
    # Compute FFT power spectrum
    fft = np.fft.fft(signal)
    power = np.abs(fft) ** 2

    # Normalize to probability distribution
    power_norm = power / np.sum(power)

    # Shannon entropy, skip zero probabilities
    entropy = -np.sum(
        power_norm[power_norm > 1e-10] * np.log(power_norm[power_norm > 1e-10])
    )

    # Normalize to [0, 1]
    max_entropy = np.log(len(signal))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def permutation_entropy(signal, embedding_dim=3):
    """Compute normalized permutation entropy.

    Measures temporal complexity via ordinal patterns (0=regular, 1=chaotic).
    """
    if len(signal) < embedding_dim:
        return 0.0

    # Count ordinal patterns
    pattern_counts = {}

    for i in range(len(signal) - embedding_dim + 1):
        window = signal[i : i + embedding_dim]
        # Get ordinal pattern: ranking of values
        pattern = tuple(np.argsort(window))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Compute Shannon entropy of pattern distribution
    total_patterns = len(signal) - embedding_dim + 1
    entropy = 0.0

    for count in pattern_counts.values():
        p = count / total_patterns
        if p > 1e-10:
            entropy -= p * np.log(p)

    # Normalize to [0, 1]
    max_entropy = np.log(np.math.factorial(embedding_dim))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def sample_entropy(signal, embedding_dim=2, tolerance=None):
    """Compute sample entropy.

    Measures self-similarity (lower=regular, higher=complex).
    Default tolerance = 0.2 * standard deviation.
    """
    if len(signal) < embedding_dim + 1:
        return 0.0

    if tolerance is None:
        tolerance = 0.2 * np.std(signal)

    def count_matches(signal, m, r):
        """Count template matches within tolerance r."""
        count = 0
        for i in range(len(signal) - m):
            for j in range(i + 1, len(signal) - m):
                template_i = signal[i : i + m]
                template_j = signal[j : j + m]
                # Chebyshev distance
                dist = np.max(np.abs(template_i - template_j))
                if dist <= r:
                    count += 1
        return count

    # Count matches for dimension m and m+1
    count_m = count_matches(signal, embedding_dim, tolerance)
    count_m1 = count_matches(signal, embedding_dim + 1, tolerance)

    # SampEn = -ln(count_m+1 / count_m)
    if count_m > 0 and count_m1 > 0:
        return -np.log(count_m1 / count_m)
    else:
        return 0.0


def compute_entropy_metrics(signal, embedding_dim=3):
    """Compute all entropy metrics for a signal.

    Args:
        signal: Input signal array
        embedding_dim: Embedding dimension for permutation entropy

    Returns:
        Dictionary with 'spectral', 'permutation', and 'sample' keys
    """
    return {
        "spectral": spectral_entropy(signal),
        "permutation": permutation_entropy(signal, embedding_dim),
        "sample": sample_entropy(signal, 2),
    }


def main():
    """Create comprehensive entropy visualization."""

    print("Entropy Metrics Visualization Example")
    print("=" * 50)

    # Generate test signals
    print("\nGenerating test signals...")
    signals = {
        "Pure Sine": generate_signal("sine"),
        "White Noise": generate_signal("noise"),
        "Chirp": generate_signal("chirp"),
        "AM-FM": generate_signal("am_fm"),
    }

    # Compute entropy metrics
    print("Computing entropy metrics...")
    results = {}
    for name, signal in signals.items():
        results[name] = compute_entropy_metrics(signal)
        print(f"  {name}: processed")

    # Create comprehensive figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Entropy Metrics Analysis", fontsize=16, fontweight="bold")
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(signals)))
    signal_names = list(signals.keys())

    # ========== Plot 1: Time Domain Signals ==========
    ax_time = fig.add_subplot(gs[0, :])
    for i, (name, signal) in enumerate(signals.items()):
        # Plot first 200 samples for clarity
        samples = min(200, len(signal))
        time_axis = np.linspace(0, 0.2, samples)
        ax_time.plot(
            time_axis, signal[:samples], label=name, color=colors[i], linewidth=1.5
        )

    ax_time.set_title("Test Signals (first 200 samples)", fontweight="bold")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.legend(loc="upper right", ncol=4)
    ax_time.grid(True, alpha=0.3)

    # ========== Plot 2: Spectral Entropy ==========
    ax_spectral = fig.add_subplot(gs[1, 0])
    spectral_values = [results[name]["spectral"] for name in signal_names]
    bars_s = ax_spectral.bar(
        range(len(signal_names)),
        spectral_values,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars_s, spectral_values)):
        ax_spectral.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax_spectral.set_title(
        "Spectral Entropy\n(0=narrowband, 1=broadband)", fontweight="bold"
    )
    ax_spectral.set_ylabel("Normalized Entropy [0, 1]")
    ax_spectral.set_ylim(0, 1.1)
    ax_spectral.set_xticks(range(len(signal_names)))
    ax_spectral.set_xticklabels(signal_names, rotation=15, ha="right")
    ax_spectral.grid(True, alpha=0.3, axis="y")

    # ========== Plot 3: Permutation Entropy ==========
    ax_permutation = fig.add_subplot(gs[1, 1])
    perm_values = [results[name]["permutation"] for name in signal_names]
    bars_p = ax_permutation.bar(
        range(len(signal_names)),
        perm_values,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars_p, perm_values)):
        ax_permutation.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax_permutation.set_title(
        "Permutation Entropy\n(0=regular, 1=chaotic)", fontweight="bold"
    )
    ax_permutation.set_ylabel("Normalized Entropy [0, 1]")
    ax_permutation.set_ylim(0, 1.1)
    ax_permutation.set_xticks(range(len(signal_names)))
    ax_permutation.set_xticklabels(signal_names, rotation=15, ha="right")
    ax_permutation.grid(True, alpha=0.3, axis="y")

    # ========== Plot 4: Sample Entropy ==========
    ax_sample = fig.add_subplot(gs[2, 0])
    sample_values = [results[name]["sample"] for name in signal_names]
    bars_sa = ax_sample.bar(
        range(len(signal_names)),
        sample_values,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars_sa, sample_values)):
        ax_sample.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax_sample.set_title(
        "Sample Entropy\n(lower=regular, higher=complex)", fontweight="bold"
    )
    ax_sample.set_ylabel("Entropy Value")
    ax_sample.set_xticks(range(len(signal_names)))
    ax_sample.set_xticklabels(signal_names, rotation=15, ha="right")
    ax_sample.grid(True, alpha=0.3, axis="y")

    # ========== Plot 5: Entropy Profile Comparison ==========
    ax_profile = fig.add_subplot(gs[2, 1])

    x_pos = np.arange(len(signal_names))
    width = 0.25

    # Normalize sample entropy to [0, 1] for comparison
    sample_normalized = (
        np.array(sample_values) / max(sample_values)
        if max(sample_values) > 0
        else np.array(sample_values)
    )

    bars1 = ax_profile.bar(
        x_pos - width,
        spectral_values,
        width,
        label="Spectral",
        color="#FF9999",
        edgecolor="black",
    )
    bars2 = ax_profile.bar(
        x_pos,
        perm_values,
        width,
        label="Permutation",
        color="#66B2FF",
        edgecolor="black",
    )
    bars3 = ax_profile.bar(
        x_pos + width,
        sample_normalized,
        width,
        label="Sample (norm)",
        color="#99FF99",
        edgecolor="black",
    )

    ax_profile.set_title(
        "Entropy Profile Comparison\n(All normalized to [0, 1])", fontweight="bold"
    )
    ax_profile.set_ylabel("Normalized Entropy")
    ax_profile.set_ylim(0, 1.2)
    ax_profile.set_xticks(x_pos)
    ax_profile.set_xticklabels(signal_names, rotation=15, ha="right")
    ax_profile.legend(loc="upper right", fontsize=9)
    ax_profile.grid(True, alpha=0.3, axis="y")

    # Save figure
    print("\nSaving visualization...")
    plt.savefig("entropy_visualization.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: entropy_visualization.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("ENTROPY METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Signal':<15} {'Spectral':<15} {'Permutation':<15} {'Sample':<15}")
    print("-" * 70)

    for name in signal_names:
        se = results[name]["spectral"]
        pe = results[name]["permutation"]
        sae = results[name]["sample"]
        print(f"{name:<15} {se:<15.4f} {pe:<15.4f} {sae:<15.4f}")

    print("=" * 70)

    # Interpretation guide
    print("\nINTERPRETATION GUIDE")
    print("-" * 70)
    print("\nSpectral Entropy (measures frequency disorder):")
    print("  < 0.3: Narrowband signal (concentrated energy)")
    print("  0.3-0.6: Mixed content")
    print("  > 0.6: Broadband signal (distributed energy)")

    print("\nPermutation Entropy (measures temporal complexity):")
    print("  < 0.3: Regular/periodic signal")
    print("  0.3-0.6: Mixed regular/chaotic")
    print("  > 0.6: Chaotic/random signal")

    print("\nSample Entropy (measures self-similarity):")
    print("  < 0.5: Highly regular")
    print("  0.5-1.0: Moderate complexity")
    print("  > 1.0: High complexity/randomness")

    print("\nSignal Analysis:")
    for name in signal_names:
        se = results[name]["spectral"]
        pe = results[name]["permutation"]
        sae = results[name]["sample"]

        characteristics = []

        if se < 0.3:
            characteristics.append("narrowband")
        elif se > 0.6:
            characteristics.append("broadband")

        if pe < 0.3:
            characteristics.append("regular")
        elif pe > 0.6:
            characteristics.append("chaotic")

        if sae < 0.5:
            characteristics.append("predictable")
        elif sae > 1.0:
            characteristics.append("complex")

        char_str = (
            ", ".join(characteristics)
            if characteristics
            else "balanced characteristics"
        )
        print(f"\n  {name}: {char_str}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
