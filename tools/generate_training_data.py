#!/usr/bin/env python3
"""
Generate synthetic training data for LSTM boundary prediction model.

Signal types:
1. Pure Tones (200 signals) - Single frequency sine waves, highly stationary
2. Chirps (200 signals) - Frequency sweeps, non-stationary with predictable boundaries
3. AM/FM Modulated (200 signals) - Amplitude and frequency modulation
4. White Noise + Signals (200 signals) - Signal buried in noise
5. Frequency Sweeps (100 signals) - Multi-octave transitions, abrupt spectral changes
6. Intermittent Signals (100 signals) - On/off switching, true intermittency

Output: NumPy compressed format (NPZ) with metadata
"""

import argparse
import numpy as np
from typing import Tuple, List
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SyntheticSignalGenerator:
    """Generate synthetic signals for LSTM training."""

    def __init__(self, sample_rate: int = 1000, duration: float = 1.0, seed: int = 42):
        """
        Initialize signal generator.

        Args:
            sample_rate: Sampling rate in Hz (default: 1000 Hz)
            duration: Signal duration in seconds (default: 1.0 s)
            seed: Random seed for reproducibility
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.t = np.linspace(0, duration, self.n_samples, endpoint=False)
        np.random.seed(seed)

    def generate_pure_tones(
        self, n_signals: int = 200
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate pure tone signals.

        Pure tones are highly stationary sine waves.

        Args:
            n_signals: Number of signals to generate

        Returns:
            List of (signal, stationarity_score, category) tuples
        """
        logger.info(f"Generating {n_signals} pure tone signals...")
        signals = []

        for i in range(n_signals):
            # Random frequency between 10 Hz and 1000 Hz
            freq = np.random.uniform(10, 1000)

            # Generate sine wave
            signal = np.sin(2 * np.pi * freq * self.t).astype(np.float32)

            # Pure tones are highly stationary
            stationarity_score = 0.95

            signals.append((signal, stationarity_score, "pure_tone"))

        logger.info(f"✓ Generated {len(signals)} pure tone signals")
        return signals

    def generate_chirps(
        self, n_signals: int = 200
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate chirp (frequency sweep) signals.

        Chirps are non-stationary with predictable boundary effects.

        Args:
            n_signals: Number of signals to generate

        Returns:
            List of (signal, stationarity_score, category) tuples
        """
        logger.info(f"Generating {n_signals} chirp signals...")
        signals = []

        for i in range(n_signals):
            # Random frequency sweep
            f_start = np.random.uniform(10, 100)
            f_end = np.random.uniform(100, 500)

            # Instantaneous frequency increases linearly with time
            phase = (
                2
                * np.pi
                * (
                    f_start * self.t
                    + (f_end - f_start) * self.t**2 / (2 * self.duration)
                )
            )
            signal = np.sin(phase).astype(np.float32)

            # Chirps are moderately stationary (non-stationary but predictable)
            stationarity_score = 0.55

            signals.append((signal, stationarity_score, "chirp"))

        logger.info(f"✓ Generated {len(signals)} chirp signals")
        return signals

    def generate_am_fm_modulated(
        self, n_signals: int = 200
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate AM/FM (amplitude/frequency) modulated signals.

        Args:
            n_signals: Number of signals to generate

        Returns:
            List of (signal, stationarity_score, category) tuples
        """
        logger.info(f"Generating {n_signals} AM/FM modulated signals...")
        signals = []

        for i in range(n_signals):
            # Carrier frequency
            carrier_freq = np.random.uniform(50, 200)

            # Modulation frequency (much lower than carrier)
            mod_freq = np.random.uniform(2, 10)

            # AM: amplitude modulation
            am_factor = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * self.t)

            # FM: frequency modulation (frequency deviation)
            freq_deviation = np.random.uniform(5, 30)
            phase = (
                2
                * np.pi
                * (
                    carrier_freq * self.t
                    + (freq_deviation / (2 * np.pi * mod_freq))
                    * np.sin(2 * np.pi * mod_freq * self.t)
                )
            )

            # Combined AM-FM signal
            signal = (am_factor * np.sin(phase)).astype(np.float32)

            # AM/FM signals are intermittently stationary
            stationarity_score = 0.45

            signals.append((signal, stationarity_score, "am_fm_modulated"))

        logger.info(f"✓ Generated {len(signals)} AM/FM modulated signals")
        return signals

    def generate_noise_plus_signal(
        self, n_signals: int = 200
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate signals with white noise added.

        Signal buried in noise with boundary effects amplified.

        Args:
            n_signals: Number of signals to generate

        Returns:
            List of (signal, stationarity_score, category) tuples
        """
        logger.info(f"Generating {n_signals} noise + signal signals...")
        signals = []

        for i in range(n_signals):
            # Random signal frequency
            sig_freq = np.random.uniform(10, 100)

            # Generate signal
            clean_signal = np.sin(2 * np.pi * sig_freq * self.t)

            # Add white noise
            snr_db = np.random.uniform(0, 10)  # SNR between 0-10 dB
            snr_linear = 10 ** (snr_db / 10)
            noise = np.random.normal(0, 1 / np.sqrt(snr_linear), self.n_samples)

            signal = (clean_signal + noise).astype(np.float32)

            # Noisy signals are less stationary
            stationarity_score = 0.50

            signals.append((signal, stationarity_score, "noise_signal"))

        logger.info(f"✓ Generated {len(signals)} noise + signal signals")
        return signals

    def generate_frequency_sweeps(
        self, n_signals: int = 100
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate multi-octave frequency sweep signals.

        Abrupt spectral changes simulate worst-case end-effect scenarios.

        Args:
            n_signals: Number of signals to generate

        Returns:
            List of (signal, stationarity_score, category) tuples
        """
        logger.info(f"Generating {n_signals} frequency sweep signals...")
        signals = []

        for i in range(n_signals):
            # Multi-octave sweep (10 Hz to 500+ Hz)
            f_start = np.random.uniform(10, 50)
            f_end = np.random.choice([200, 300, 500, 750, 1000])

            # Exponential frequency sweep (multi-octave)
            log_start = np.log2(f_start)
            log_end = np.log2(f_end)
            freq_t = 2 ** (log_start + (log_end - log_start) * self.t / self.duration)

            # Integrate frequency to get phase
            phase = 2 * np.pi * np.cumsum(freq_t) / self.sample_rate
            signal = np.sin(phase).astype(np.float32)

            # Frequency sweeps are highly non-stationary
            stationarity_score = 0.30

            signals.append((signal, stationarity_score, "frequency_sweep"))

        logger.info(f"✓ Generated {len(signals)} frequency sweep signals")
        return signals

    def generate_intermittent_signals(
        self, n_signals: int = 100
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate intermittent (on/off) signals.

        True intermittency is worst-case for stationary prediction methods.

        Args:
            n_signals: Number of signals to generate

        Returns:
            List of (signal, stationarity_score, category) tuples
        """
        logger.info(f"Generating {n_signals} intermittent signals...")
        signals = []

        for i in range(n_signals):
            signal = np.zeros(self.n_samples, dtype=np.float32)

            # Number of on/off transitions
            n_transitions = np.random.randint(2, 6)
            transition_indices = np.sort(
                np.random.choice(self.n_samples, size=n_transitions, replace=False)
            )

            # Alternate between on and off
            is_on = True
            start_idx = 0

            for idx in transition_indices:
                if is_on:
                    # Generate signal in this segment
                    freq = np.random.uniform(20, 200)
                    segment_t = np.linspace(
                        0, (idx - start_idx) / self.sample_rate, idx - start_idx
                    )
                    signal[start_idx:idx] = np.sin(2 * np.pi * freq * segment_t).astype(
                        np.float32
                    )

                is_on = not is_on
                start_idx = idx

            # Fill remaining part
            if is_on and start_idx < self.n_samples:
                freq = np.random.uniform(20, 200)
                segment_t = np.linspace(
                    0,
                    (self.n_samples - start_idx) / self.sample_rate,
                    self.n_samples - start_idx,
                )
                signal[start_idx:] = np.sin(2 * np.pi * freq * segment_t).astype(
                    np.float32
                )

            # Intermittent signals have lowest stationarity
            stationarity_score = 0.25

            signals.append((signal, stationarity_score, "intermittent"))

        logger.info(f"✓ Generated {len(signals)} intermittent signals")
        return signals

    def apply_augmentation(
        self, signal: np.ndarray, stationarity_score: float
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Apply data augmentation to a signal.

        Augmentations:
        - Noise injection (small amount)
        - Time scaling (slight speed changes)
        - Amplitude scaling

        Args:
            signal: Input signal
            stationarity_score: Original stationarity score

        Returns:
            List of augmented (signal, stationarity_score) tuples
        """
        augmented = [(signal, stationarity_score)]

        # Noise injection variant
        noise_level = np.random.uniform(0.01, 0.05)
        noisy = signal + noise_level * np.random.normal(0, 1, len(signal))
        augmented.append((noisy.astype(np.float32), stationarity_score * 0.95))

        # Amplitude scaling variant
        amp_scale = np.random.uniform(0.8, 1.2)
        scaled = amp_scale * signal
        augmented.append((scaled.astype(np.float32), stationarity_score))

        return augmented


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for LSTM boundary prediction model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.npz",
        help="Output file path (default: training_data.npz)",
    )
    parser.add_argument(
        "--n_signals",
        type=int,
        default=1000,
        help="Total number of signals to generate (default: 1000)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1000,
        help="Sampling rate in Hz (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation (increases dataset size)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Synthetic Training Data Generator")
    logger.info("=" * 70)
    logger.info(f"Output file: {args.output}")
    logger.info(f"Total signals: {args.n_signals}")
    logger.info(f"Sampling rate: {args.sample_rate} Hz")
    logger.info(f"Augmentation: {'Enabled' if args.augment else 'Disabled'}")
    logger.info("")

    # Initialize generator
    gen = SyntheticSignalGenerator(sample_rate=args.sample_rate, seed=args.seed)

    # Generate all signal types
    all_signals = []
    all_stationarity_scores = []
    all_categories = []

    # Pure tones: 200 signals
    signals = gen.generate_pure_tones(n_signals=200)
    all_signals.extend([s[0] for s in signals])
    all_stationarity_scores.extend([s[1] for s in signals])
    all_categories.extend([s[2] for s in signals])

    # Chirps: 200 signals
    signals = gen.generate_chirps(n_signals=200)
    all_signals.extend([s[0] for s in signals])
    all_stationarity_scores.extend([s[1] for s in signals])
    all_categories.extend([s[2] for s in signals])

    # AM/FM modulated: 200 signals
    signals = gen.generate_am_fm_modulated(n_signals=200)
    all_signals.extend([s[0] for s in signals])
    all_stationarity_scores.extend([s[1] for s in signals])
    all_categories.extend([s[2] for s in signals])

    # White noise + signal: 200 signals
    signals = gen.generate_noise_plus_signal(n_signals=200)
    all_signals.extend([s[0] for s in signals])
    all_stationarity_scores.extend([s[1] for s in signals])
    all_categories.extend([s[2] for s in signals])

    # Frequency sweeps: 100 signals
    signals = gen.generate_frequency_sweeps(n_signals=100)
    all_signals.extend([s[0] for s in signals])
    all_stationarity_scores.extend([s[1] for s in signals])
    all_categories.extend([s[2] for s in signals])

    # Intermittent signals: 100 signals
    signals = gen.generate_intermittent_signals(n_signals=100)
    all_signals.extend([s[0] for s in signals])
    all_stationarity_scores.extend([s[1] for s in signals])
    all_categories.extend([s[2] for s in signals])

    logger.info("")
    logger.info(f"Total signals generated: {len(all_signals)}")

    # Apply augmentation if requested
    if args.augment:
        logger.info("\nApplying data augmentation...")
        augmented_signals = []
        augmented_scores = []
        augmented_categories = []

        for sig, score, cat in zip(
            all_signals, all_stationarity_scores, all_categories
        ):
            augmented_variants = gen.apply_augmentation(sig, score)
            augmented_signals.extend([v[0] for v in augmented_variants])
            augmented_scores.extend([v[1] for v in augmented_variants])
            augmented_categories.extend([cat] * len(augmented_variants))

        all_signals = augmented_signals
        all_stationarity_scores = augmented_scores
        all_categories = augmented_categories
        logger.info(f"After augmentation: {len(all_signals)} signals (3x increase)")

    # Convert to NumPy arrays
    signals_array = np.array(all_signals, dtype=np.float32)
    stationarity_array = np.array(all_stationarity_scores, dtype=np.float32)
    categories_array = np.array(all_categories, dtype=object)

    # Create labels: 0 = stationary (score > 0.7), 1 = non-stationary
    boundary_labels = (stationarity_array <= 0.7).astype(np.int32)

    logger.info("")
    logger.info(f"Signal shape: {signals_array.shape}")
    logger.info(f"Stationarity scores shape: {stationarity_array.shape}")
    logger.info(f"Boundary labels shape: {boundary_labels.shape}")
    logger.info("")
    logger.info("Stationarity Score Distribution:")
    logger.info(f"  Min: {stationarity_array.min():.3f}")
    logger.info(f"  Max: {stationarity_array.max():.3f}")
    logger.info(f"  Mean: {stationarity_array.mean():.3f}")
    logger.info(f"  Stationary (>0.7): {(stationarity_array > 0.7).sum()} signals")
    logger.info(f"  Non-stationary (≤0.7): {(stationarity_array <= 0.7).sum()} signals")

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        signals=signals_array,
        stationarity_scores=stationarity_array,
        boundary_labels=boundary_labels,
        categories=categories_array,
        sample_rate=np.array([args.sample_rate], dtype=np.int32),
        seed=np.array([args.seed], dtype=np.int32),
    )

    logger.info("")
    logger.info(f"✓ Dataset saved to: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training data generation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
