"""
Generate synthetic ferromode_training signals for LSTM boundary prediction model.

Generates diverse signal types suitable for ferromode_training the boundary prediction
neural network: pure tones, chirps, AM/FM modulated, non-stationary, bursts,
and real-world-like signals.
"""

import numpy as np
from typing import Tuple, List


def generate_synthetic_signals(
    num_signals: int = 1000,
    signal_length: int = 500,
    random_seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate diverse synthetic signals for ferromode_training.

    Args:
        num_signals: Number of ferromode_training signals to generate (default: 1000)
        signal_length: Length of each signal in samples (default: 500)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        (signals, targets) tuples where:
        - signals: list of signal chunks (numpy arrays)
        - targets: list of ground-truth extensions (numpy arrays)
    """
    np.random.seed(random_seed)

    signals = []
    targets = []

    # Generate different signal types (1000 signals / 6 types ≈ 167 each)
    signals_per_type = num_signals // 6

    # 1. Pure tones (10%)
    print(f"[1/6] Generating {signals_per_type} pure tones...")
    for _ in range(signals_per_type):
        freq = np.random.uniform(0.05, 0.5)
        t = np.arange(signal_length)
        sig = np.sin(2 * np.pi * freq * t)
        signals.append(sig)
        # For pure tone, predict continuation
        target = np.sin(2 * np.pi * freq * (t[-10:] + 10))
        targets.append(target)

    # 2. Chirps (15%)
    print(f"[2/6] Generating {signals_per_type} chirps...")
    for _ in range(signals_per_type):
        f0 = np.random.uniform(0.05, 0.2)
        f1 = np.random.uniform(0.2, 0.5)
        t = np.arange(signal_length) / signal_length
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2)
        sig = np.sin(phase)
        signals.append(sig)
        # Predict chirp continuation
        t_future = (np.arange(signal_length, signal_length + 10)) / signal_length
        phase_future = 2 * np.pi * (f0 * t_future + (f1 - f0) * t_future**2 / 2)
        target = np.sin(phase_future)
        targets.append(target)

    # 3. AM/FM modulated (15%)
    print(f"[3/6] Generating {signals_per_type} AM/FM modulated signals...")
    for _ in range(signals_per_type):
        carrier_freq = np.random.uniform(0.15, 0.35)
        mod_freq = np.random.uniform(0.05, 0.15)
        depth = np.random.uniform(0.5, 0.9)

        t = np.arange(signal_length)
        envelope = 1 + depth * np.sin(2 * np.pi * mod_freq * t / signal_length)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        sig = envelope * carrier
        signals.append(sig)

        # Predict continuation
        t_future = np.arange(signal_length, signal_length + 10)
        env_future = 1 + depth * np.sin(2 * np.pi * mod_freq * t_future / signal_length)
        car_future = np.sin(2 * np.pi * carrier_freq * t_future)
        target = env_future * car_future
        targets.append(target)

    # 4. Non-stationary (frequency sweep) (20%)
    print(f"[4/6] Generating {signals_per_type} non-stationary signals...")
    for _ in range(signals_per_type):
        # Multiple frequency components with time-varying amplitudes
        freqs = np.random.uniform(0.05, 0.4, 3)
        amps = np.random.uniform(0.3, 1.0, 3)

        t = np.arange(signal_length)
        sig = np.zeros(signal_length)
        for freq, amp in zip(freqs, amps):
            # Time-varying amplitude
            amp_envelope = amp * (1 + 0.5 * np.sin(2 * np.pi * 0.01 * t))
            sig += amp_envelope * np.sin(2 * np.pi * freq * t)

        signals.append(sig / 3)  # Normalize

        # Predict continuation with same frequency components
        t_future = np.arange(signal_length, signal_length + 10)
        target = np.zeros(10)
        for freq, amp in zip(freqs, amps):
            amp_env = amp * (1 + 0.5 * np.sin(2 * np.pi * 0.01 * t_future))
            target += amp_env * np.sin(2 * np.pi * freq * t_future)
        targets.append(target / 3)

    # 5. Intermittent/Burst signals (20%)
    print(f"[5/6] Generating {signals_per_type} intermittent signals...")
    for _ in range(signals_per_type):
        sig = np.zeros(signal_length)
        num_bursts = np.random.randint(2, 4)

        for _ in range(num_bursts):
            # Ensure burst fits safely within signal
            max_start = signal_length - 30
            if max_start > 0:
                burst_start = np.random.randint(0, max_start)
                burst_len = np.random.randint(20, 30)  # Keep small
                burst_freq = np.random.uniform(0.1, 0.4)

                # Ensure we don't overflow
                end_idx = min(burst_start + burst_len, signal_length)
                actual_len = end_idx - burst_start

                burst_t = np.arange(actual_len)
                burst = np.sin(2 * np.pi * burst_freq * burst_t)
                sig[burst_start:end_idx] += burst

        sig = np.tanh(sig)  # Limit amplitude
        signals.append(sig)

        # Predict continuation (probably quiet, unless burst continues)
        target = np.random.normal(0, 0.1, 10)
        targets.append(target)

    # 6. Real-world-like signals (20%)
    print(f"[6/6] Generating {signals_per_type} complex real-world-like signals...")
    for _ in range(signals_per_type):
        # Mix of multiple frequencies with noise
        t = np.arange(signal_length)
        sig = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        sig += 0.3 * np.sin(2 * np.pi * 0.2 * t)
        sig += 0.2 * np.sin(2 * np.pi * 0.3 * t)
        sig += 0.1 * np.random.randn(signal_length)
        signals.append(sig)

        # Predict continuation
        t_future = np.arange(signal_length, signal_length + 10)
        target = 0.5 * np.sin(2 * np.pi * 0.1 * t_future)
        target += 0.3 * np.sin(2 * np.pi * 0.2 * t_future)
        target += 0.2 * np.sin(2 * np.pi * 0.3 * t_future)
        target += 0.05 * np.random.randn(10)
        targets.append(target)

    print(f"\n✓ Generated {len(signals)} ferromode_training signals")
    return signals, targets
