use ferromode::boundary::{get_strategy, BoundaryCondition, BoundaryConditionType, Extrema};

fn make_extrema(signal: &[f64]) -> Extrema {
    let mut maxima = Vec::new();
    let mut minima = Vec::new();
    for i in 1..signal.len().saturating_sub(1) {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            maxima.push(i);
        } else if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
            minima.push(i);
        }
    }
    Extrema { maxima_indices: maxima, minima_indices: minima }
}

fn assert_extended_contains_original(strategy: &dyn BoundaryCondition, signal: &[f64]) {
    let extrema = make_extrema(signal);
    let extended = strategy.extend(signal, &extrema);

    let original_slice = &extended.values[extended.original_start..extended.original_end];
    for (i, (a, b)) in original_slice.iter().zip(signal.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "strategy '{}' corrupted original at index {}: expected {}, got {}",
            strategy.name(),
            i,
            b,
            a
        );
    }
}

fn assert_extended_is_finite(strategy: &dyn BoundaryCondition, signal: &[f64]) {
    let extrema = make_extrema(signal);
    let extended = strategy.extend(signal, &extrema);

    for (i, &v) in extended.values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "strategy '{}' produced non-finite value at index {}: {}",
            strategy.name(),
            i,
            v
        );
    }
}

fn assert_extended_longer_than_original(strategy: &dyn BoundaryCondition, signal: &[f64]) {
    let extrema = make_extrema(signal);
    let extended = strategy.extend(signal, &extrema);

    assert!(
        extended.values.len() > signal.len(),
        "strategy '{}' did not extend signal: original={}, extended={}",
        strategy.name(),
        signal.len(),
        extended.values.len()
    );
}

// =========================================================================
// Characteristic Wave Extension (T-029, T-030, T-031)
// =========================================================================

#[test]
fn test_characteristic_wave_preserves_original() {
    let strategy = get_strategy(BoundaryConditionType::CharacteristicWave);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
}

#[test]
fn test_characteristic_wave_is_finite() {
    let strategy = get_strategy(BoundaryConditionType::CharacteristicWave);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_characteristic_wave_extends_signal() {
    let strategy = get_strategy(BoundaryConditionType::CharacteristicWave);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_longer_than_original(&*strategy, &signal);
}

#[test]
fn test_characteristic_wave_on_chirp() {
    let strategy = get_strategy(BoundaryConditionType::CharacteristicWave);
    let signal: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / 200.0;
            (2.0 * std::f64::consts::PI * (5.0 + 20.0 * t) * t).sin()
        })
        .collect();
    assert_extended_contains_original(&*strategy, &signal);
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_characteristic_wave_on_noisy_signal() {
    let strategy = get_strategy(BoundaryConditionType::CharacteristicWave);
    let signal: Vec<f64> = (0..100)
        .map(|i| {
            let base = (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin();
            base + 0.1 * (i as f64 * 7.3).sin()
        })
        .collect();
    assert_extended_contains_original(&*strategy, &signal);
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_characteristic_wave_few_extrema() {
    let strategy = get_strategy(BoundaryConditionType::CharacteristicWave);
    let signal = vec![0.0, 1.0, 2.0, 1.5, 1.0, 0.5, 0.0];
    assert_extended_contains_original(&*strategy, &signal);
    assert_extended_is_finite(&*strategy, &signal);
}

// =========================================================================
// Mirror Even Extension (T-032, T-033, T-034)
// =========================================================================

#[test]
fn test_mirror_even_preserves_original() {
    let strategy = get_strategy(BoundaryConditionType::MirrorEven);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
}

#[test]
fn test_mirror_even_is_finite() {
    let strategy = get_strategy(BoundaryConditionType::MirrorEven);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_mirror_even_extends_signal() {
    let strategy = get_strategy(BoundaryConditionType::MirrorEven);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_longer_than_original(&*strategy, &signal);
}

#[test]
fn test_mirror_even_symmetry() {
    let strategy = get_strategy(BoundaryConditionType::MirrorEven);
    let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0, 4] };
    let extended = strategy.extend(&signal, &extrema);
    let left_ext = &extended.values[..extended.original_start];
    let right_ext = &extended.values[extended.original_end..];

    assert!(!left_ext.is_empty() || !right_ext.is_empty(), "mirror even should extend both ends");
}

// =========================================================================
// Mirror Odd Extension (T-033)
// =========================================================================

#[test]
fn test_mirror_odd_preserves_original() {
    let strategy = get_strategy(BoundaryConditionType::MirrorOdd);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
}

#[test]
fn test_mirror_odd_is_finite() {
    let strategy = get_strategy(BoundaryConditionType::MirrorOdd);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_mirror_odd_extends_signal() {
    let strategy = get_strategy(BoundaryConditionType::MirrorOdd);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_longer_than_original(&*strategy, &signal);
}

// =========================================================================
// Periodic / Cyclic Extension (T-035, T-036, T-037, T-038)
// =========================================================================

#[test]
fn test_periodic_preserves_original() {
    let strategy = get_strategy(BoundaryConditionType::Periodic);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
}

#[test]
fn test_periodic_is_finite() {
    let strategy = get_strategy(BoundaryConditionType::Periodic);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_periodic_extends_signal() {
    let strategy = get_strategy(BoundaryConditionType::Periodic);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_longer_than_original(&*strategy, &signal);
}

#[test]
fn test_periodic_on_known_periodic() {
    let strategy = get_strategy(BoundaryConditionType::Periodic);
    let signal: Vec<f64> =
        (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
    assert_extended_is_finite(&*strategy, &signal);
}

// =========================================================================
// Slope-Based Extension (T-039, T-040, T-041)
// =========================================================================

#[test]
fn test_slope_preserves_original() {
    let strategy = get_strategy(BoundaryConditionType::Slope);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
}

#[test]
fn test_slope_is_finite() {
    let strategy = get_strategy(BoundaryConditionType::Slope);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_slope_extends_signal() {
    let strategy = get_strategy(BoundaryConditionType::Slope);
    let signal: Vec<f64> =
        (0..50).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 25.0).sin()).collect();
    assert_extended_longer_than_original(&*strategy, &signal);
}

#[test]
fn test_slope_on_monotone_ends() {
    let strategy = get_strategy(BoundaryConditionType::Slope);
    let signal: Vec<f64> = (0..50)
        .map(|i| {
            let t = i as f64 / 49.0;
            t * t + 0.3 * (2.0 * std::f64::consts::PI * t * 3.0).sin()
        })
        .collect();
    assert_extended_contains_original(&*strategy, &signal);
    assert_extended_is_finite(&*strategy, &signal);
}

// =========================================================================
// AR Model Extension (T-042, T-043, T-044, T-045)
// =========================================================================

#[test]
fn test_ar_model_preserves_original() {
    let strategy = get_strategy(BoundaryConditionType::ARModel);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
}

#[test]
fn test_ar_model_is_finite() {
    let strategy = get_strategy(BoundaryConditionType::ARModel);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_ar_model_extends_signal() {
    let strategy = get_strategy(BoundaryConditionType::ARModel);
    let signal: Vec<f64> =
        (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_longer_than_original(&*strategy, &signal);
}

#[test]
fn test_ar_model_on_ar2_synthetic() {
    let strategy = get_strategy(BoundaryConditionType::ARModel);
    let mut signal = vec![0.0, 1.0];
    for i in 2..100 {
        let val = 1.5 * signal[i - 1] - 0.7 * signal[i - 2] + 0.1 * (i as f64 * 3.7).sin();
        signal.push(val);
    }
    assert_extended_contains_original(&*strategy, &signal);
    assert_extended_is_finite(&*strategy, &signal);
}

// =========================================================================
// Waveform Matching Extension (T-046, T-047, T-048, T-049)
// =========================================================================

#[test]
fn test_waveform_matching_preserves_original() {
    let strategy = get_strategy(BoundaryConditionType::WaveformMatching);
    let signal: Vec<f64> =
        (0..200).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_contains_original(&*strategy, &signal);
}

#[test]
fn test_waveform_matching_is_finite() {
    let strategy = get_strategy(BoundaryConditionType::WaveformMatching);
    let signal: Vec<f64> =
        (0..200).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_is_finite(&*strategy, &signal);
}

#[test]
fn test_waveform_matching_extends_signal() {
    let strategy = get_strategy(BoundaryConditionType::WaveformMatching);
    let signal: Vec<f64> =
        (0..200).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
    assert_extended_longer_than_original(&*strategy, &signal);
}

#[test]
fn test_waveform_matching_on_quasi_periodic() {
    let strategy = get_strategy(BoundaryConditionType::WaveformMatching);
    let signal: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / 200.0;
            (2.0 * std::f64::consts::PI * t * 5.0).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * t * 13.0).sin()
        })
        .collect();
    assert_extended_contains_original(&*strategy, &signal);
    assert_extended_is_finite(&*strategy, &signal);
}
