#!/usr/bin/env python3
"""
Streaming decomposition example using Python bindings.

Demonstrates chunk-by-chunk signal decomposition with metrics tracking.
"""

import sys
import math

# Try importing the ferromode_py module
try:
    import ferromode_py
except ImportError:
    print("Error: ferromode_py module not found.")
    print("Please build the Python bindings first:")
    print("  cd crates/ferromode-py && pip install -e .")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Install with: pip install numpy")
    sys.exit(1)


def main():
    print("=== Streaming Decomposition: Python Example ===\n")

    # Create decomposer
    print("Creating StreamingDecomposer...")
    decomposer = ferromode_py.StreamingDecomposer(
        max_imfs=8, chunk_size=512, buffer_size=4096, boundary_condition="mirror"
    )
    print("✓ Decomposer created\n")

    # Parameters
    chunk_size = 512
    num_chunks = 10

    print(f"Configuration:")
    print(f"  Max IMFs: 8")
    print(f"  Chunk size: {chunk_size} samples")
    print(f"  Number of chunks: {num_chunks}\n")

    # Process chunks
    print("Processing signal chunks...\n")

    for chunk_idx in range(num_chunks):
        # Generate signal chunk (sine wave with increasing frequency)
        t = np.arange(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size)
        freq = 0.02 + chunk_idx * 0.001
        chunk = np.sin(2 * np.pi * freq * t / 1024).astype(np.float64)

        # Decompose
        try:
            result = decomposer.decompose_chunk(chunk)
        except Exception as e:
            print(f"Error decomposing chunk {chunk_idx}: {e}")
            continue

        # Display results
        imfs = result.get("imfs")
        residue = result.get("residue")
        metrics = result.get("metrics", {})

        n_imfs = imfs.shape[0] if imfs is not None else 0
        n_samples = residue.shape[0] if residue is not None else 0

        print(f"Chunk {chunk_idx:2d}:")
        print(f"  IMFs extracted: {n_imfs}")
        print(f"  Residue shape: {residue.shape if residue is not None else 'None'}")

        if metrics:
            print(f"  Metrics:")
            print(f"    Spectral entropy: {metrics.get('spectral_entropy', 0):.4f}")
            print(f"    Stationarity score: {metrics.get('stationarity_score', 0):.4f}")
            print(f"    Extrema spacing CV: {metrics.get('extrema_spacing_cv', 0):.4f}")

        print()

    print("✓ Processing complete!")

    # Test reset
    print("\nTesting reset...")
    decomposer.reset()
    print("✓ Decomposer reset successfully")


if __name__ == "__main__":
    main()
