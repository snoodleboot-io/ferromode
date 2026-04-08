#![cfg(test)]
#![allow(clippy::cast_possible_truncation)]

//! Numerical parity validation tests for GPU vs CPU execution.
//!
//! Verifies that GPU and CPU implementations produce identical results
//! within acceptable numerical tolerance (< 1e-5 relative error).

use crate::adapters::gpu::{EnsembleExecutor, ExecutorConfig};
use crate::algorithms::ceemdan::ceemdan;
use crate::algorithms::eemd::eemd;
use crate::algorithms::eemd::EnsembleConfig;
use crate::algorithms::emd::EmdConfig;
use crate::types::Signal;

// ---------------------------------------------------------------------------
// Utility: Signal generation
// ---------------------------------------------------------------------------

/// Generate synthetic test signal.
fn synthetic_signal(samples: usize) -> Vec<f64> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / samples as f64 * 4.0 * std::f64::consts::PI;
            let low_freq = (t).sin();
            let high_freq = 0.3 * (5.0 * t).sin();
            low_freq + high_freq
        })
        .collect()
}

/// Generate white noise signal.
fn white_noise(samples: usize) -> Vec<f64> {
    let mut rng = 1u64;
    (0..samples)
        .map(|_| {
            rng = rng.wrapping_mul(2654435761).wrapping_add(2246822519);
            let normalized = ((rng >> 32) as f64) / (u32::MAX as f64);
            (normalized * 2.0 - 1.0) * 0.1
        })
        .collect()
}

/// Generate chirp signal.
fn chirp_signal(samples: usize, start_hz: f64, end_hz: f64) -> Vec<f64> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / samples as f64;
            let freq = start_hz + (end_hz - start_hz) * t;
            let phase = 2.0 * std::f64::consts::PI * freq * t * t;
            phase.sin()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Utility: Comparison functions
// ---------------------------------------------------------------------------

/// Compute relative error between two values.
fn relative_error(computed: f64, reference: f64) -> f64 {
    if reference.abs() < 1e-10 {
        computed.abs()
    } else {
        (computed - reference).abs() / reference.abs()
    }
}

/// Compare two IMF collections with tolerance.
fn compare_imf_collections(
    computed: &crate::types::ImfCollection,
    reference: &crate::types::ImfCollection,
    tolerance: f64,
) -> Result<(), String> {
    if computed.imfs.len() != reference.imfs.len() {
        return Err(format!(
            "IMF count mismatch: computed {} vs reference {}",
            computed.imfs.len(),
            reference.imfs.len()
        ));
    }

    for (i, (comp_imf, ref_imf)) in computed.imfs.iter().zip(&reference.imfs).enumerate() {
        if comp_imf.len() != ref_imf.len() {
            return Err(format!(
                "IMF {} length mismatch: computed {} vs reference {}",
                i,
                comp_imf.len(),
                ref_imf.len()
            ));
        }

        for (j, (&comp_val, &ref_val)) in comp_imf.iter().zip(ref_imf).enumerate() {
            let error = relative_error(comp_val, ref_val);
            if error > tolerance && ref_val.abs() > 1e-10 {
                return Err(format!(
                    "IMF {} sample {} exceeds tolerance: error {:.2e} > {:.2e}",
                    i, j, error, tolerance
                ));
            }
        }
    }

    // Check residue
    if computed.residue.len() != reference.residue.len() {
        return Err(format!(
            "Residue length mismatch: computed {} vs reference {}",
            computed.residue.len(),
            reference.residue.len()
        ));
    }

    for (j, (&comp_val, &ref_val)) in computed.residue.iter().zip(&reference.residue).enumerate() {
        let error = relative_error(comp_val, ref_val);
        if error > tolerance && ref_val.abs() > 1e-10 {
            return Err(format!(
                "Residue sample {} exceeds tolerance: error {:.2e} > {:.2e}",
                j, error, tolerance
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests: EEMD Parity
// ---------------------------------------------------------------------------

#[test]
fn test_eemd_parity_synthetic() {
    let signal_data = synthetic_signal(512);
    let config = EnsembleConfig { num_ensembles: 25, noise_std: 0.2, seed: Some(42) };
    let emd_config = EmdConfig::default();

    // CPU result
    let cpu_result = eemd(&signal_data, &config, &emd_config).expect("CPU EEMD failed");

    // GPU result (with fallback)
    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig::default();
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");
    let gpu_result = executor.execute_gpu_eemd(&signal, &config).expect("GPU EEMD failed");

    // Compare
    compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
        .expect("GPU/CPU EEMD parity failed for synthetic signal");
}

#[test]
fn test_eemd_parity_white_noise() {
    let signal_data = white_noise(512);
    let config = EnsembleConfig { num_ensembles: 25, noise_std: 0.2, seed: Some(42) };
    let emd_config = EmdConfig::default();

    let cpu_result = eemd(&signal_data, &config, &emd_config).expect("CPU EEMD failed");

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig::default();
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");
    let gpu_result = executor.execute_gpu_eemd(&signal, &config).expect("GPU EEMD failed");

    compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
        .expect("GPU/CPU EEMD parity failed for white noise");
}

#[test]
fn test_eemd_parity_chirp() {
    let signal_data = chirp_signal(512, 10.0, 100.0);
    let config = EnsembleConfig { num_ensembles: 25, noise_std: 0.2, seed: Some(42) };
    let emd_config = EmdConfig::default();

    let cpu_result = eemd(&signal_data, &config, &emd_config).expect("CPU EEMD failed");

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig::default();
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");
    let gpu_result = executor.execute_gpu_eemd(&signal, &config).expect("GPU EEMD failed");

    compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
        .expect("GPU/CPU EEMD parity failed for chirp signal");
}

#[test]
fn test_eemd_parity_various_sizes() {
    let sizes = vec![256, 512, 1024];

    for &size in &sizes {
        let signal_data = synthetic_signal(size);
        let config = EnsembleConfig { num_ensembles: 20, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let cpu_result = eemd(&signal_data, &config, &emd_config)
            .expect(&format!("CPU EEMD failed for size {}", size));

        let signal = Signal::with_sample_rate(&signal_data, 1.0)
            .expect(&format!("Signal creation failed for size {}", size));
        let executor_config = ExecutorConfig::default();
        let mut executor = EnsembleExecutor::new(executor_config)
            .expect(&format!("Executor creation failed for size {}", size));
        let gpu_result = executor
            .execute_gpu_eemd(&signal, &config)
            .expect(&format!("GPU EEMD failed for size {}", size));

        compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
            .expect(&format!("GPU/CPU EEMD parity failed for size {}", size));
    }
}

// ---------------------------------------------------------------------------
// Tests: CEEMDAN Parity
// ---------------------------------------------------------------------------

#[test]
fn test_ceemdan_parity_synthetic() {
    let signal_data = synthetic_signal(512);
    let config = EnsembleConfig { num_ensembles: 15, noise_std: 0.2, seed: Some(42) };
    let emd_config = EmdConfig::default();

    let cpu_result = ceemdan(&signal_data, &config, &emd_config).expect("CPU CEEMDAN failed");

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig::default();
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");
    let gpu_result = executor.execute_gpu_ceemdan(&signal, &config).expect("GPU CEEMDAN failed");

    compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
        .expect("GPU/CPU CEEMDAN parity failed for synthetic signal");
}

#[test]
fn test_ceemdan_parity_white_noise() {
    let signal_data = white_noise(512);
    let config = EnsembleConfig { num_ensembles: 15, noise_std: 0.2, seed: Some(42) };
    let emd_config = EmdConfig::default();

    let cpu_result = ceemdan(&signal_data, &config, &emd_config).expect("CPU CEEMDAN failed");

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig::default();
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");
    let gpu_result = executor.execute_gpu_ceemdan(&signal, &config).expect("GPU CEEMDAN failed");

    compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
        .expect("GPU/CPU CEEMDAN parity failed for white noise");
}

#[test]
fn test_ceemdan_parity_chirp() {
    let signal_data = chirp_signal(512, 10.0, 100.0);
    let config = EnsembleConfig { num_ensembles: 15, noise_std: 0.2, seed: Some(42) };
    let emd_config = EmdConfig::default();

    let cpu_result = ceemdan(&signal_data, &config, &emd_config).expect("CPU CEEMDAN failed");

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig::default();
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");
    let gpu_result = executor.execute_gpu_ceemdan(&signal, &config).expect("GPU CEEMDAN failed");

    compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
        .expect("GPU/CPU CEEMDAN parity failed for chirp signal");
}

#[test]
fn test_ceemdan_parity_various_sizes() {
    let sizes = vec![256, 512, 1024];

    for &size in &sizes {
        let signal_data = synthetic_signal(size);
        let config = EnsembleConfig { num_ensembles: 15, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let cpu_result = ceemdan(&signal_data, &config, &emd_config)
            .expect(&format!("CPU CEEMDAN failed for size {}", size));

        let signal = Signal::with_sample_rate(&signal_data, 1.0)
            .expect(&format!("Signal creation failed for size {}", size));
        let executor_config = ExecutorConfig::default();
        let mut executor = EnsembleExecutor::new(executor_config)
            .expect(&format!("Executor creation failed for size {}", size));
        let gpu_result = executor
            .execute_gpu_ceemdan(&signal, &config)
            .expect(&format!("GPU CEEMDAN failed for size {}", size));

        compare_imf_collections(&gpu_result, &cpu_result.imfs, 1e-5)
            .expect(&format!("GPU/CPU CEEMDAN parity failed for size {}", size));
    }
}

// ---------------------------------------------------------------------------
// Tests: Memory profiling
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_memory_efficiency_eemd() {
    let signal_data = synthetic_signal(10_240);
    let config = EnsembleConfig { num_ensembles: 50, noise_std: 0.2, seed: Some(42) };

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig {
        max_gpu_memory: 8 * 1024 * 1024 * 1024,
        batch_size: 16,
        profiling_enabled: true,
    };
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");

    let mem_before = executor.available_memory();
    let _ = executor.execute_gpu_eemd(&signal, &config);
    let mem_after = executor.available_memory();

    let _mem_used = mem_before.saturating_sub(mem_after);

    // Verify memory didn't exceed limit
    assert!(executor.available_memory() > 0, "GPU memory exhausted");
    assert!(executor.available_memory() <= mem_before, "Memory state invalid");
}

#[test]
fn test_gpu_memory_efficiency_ceemdan() {
    let signal_data = synthetic_signal(10_240);
    let config = EnsembleConfig { num_ensembles: 25, noise_std: 0.2, seed: Some(42) };

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig {
        max_gpu_memory: 8 * 1024 * 1024 * 1024,
        batch_size: 16,
        profiling_enabled: true,
    };
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");

    let mem_before = executor.available_memory();
    let _ = executor.execute_gpu_ceemdan(&signal, &config);
    let mem_after = executor.available_memory();

    let _mem_used = mem_before.saturating_sub(mem_after);

    // Verify memory didn't exceed limit
    assert!(executor.available_memory() > 0, "GPU memory exhausted");
    assert!(executor.available_memory() <= mem_before, "Memory state invalid");
}

// ---------------------------------------------------------------------------
// Tests: Executor fallback behavior
// ---------------------------------------------------------------------------

#[test]
fn test_executor_fallback_when_gpu_disabled() {
    let signal_data = synthetic_signal(512);
    let config = EnsembleConfig { num_ensembles: 20, noise_std: 0.2, seed: Some(42) };

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig::default();
    let executor = EnsembleExecutor::new(executor_config)
        .expect("Executor creation failed")
        .with_gpu_disabled();
    let mut executor = executor;

    // Should still produce a result even with GPU disabled (CPU fallback)
    let result = executor.execute_gpu_eemd(&signal, &config);
    assert!(result.is_ok(), "Executor fallback failed");
}

#[test]
fn test_executor_statistics() {
    let signal_data = synthetic_signal(512);
    let config = EnsembleConfig { num_ensembles: 20, noise_std: 0.2, seed: Some(42) };

    let signal = Signal::with_sample_rate(&signal_data, 1.0).expect("Signal creation failed");
    let executor_config = ExecutorConfig { profiling_enabled: true, ..Default::default() };
    let mut executor = EnsembleExecutor::new(executor_config).expect("Executor creation failed");

    let _ = executor.execute_gpu_eemd(&signal, &config);

    let stats = executor.stats();
    assert!(stats.trials_completed > 0, "No trials recorded");
    assert!(stats.total_time.as_secs_f64() > 0.0, "No timing recorded");
}
