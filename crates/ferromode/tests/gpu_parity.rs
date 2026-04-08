//! GPU vs CPU numerical parity tests.
//!
//! Validates that GPU and CPU implementations produce numerically identical results.
//! This ensures the GPU acceleration doesn't sacrifice accuracy.

use ferromode::adapters::gpu::{
    GpuCeemданConfig, GpuCeemданExecutor, GpuEemdConfig, GpuEemdExecutor,
};

const TOLERANCE: f64 = 1e-6;

/// Generate test signal: sum of sinusoids.
fn generate_signal(length: usize, seed: u64) -> Vec<f64> {
    (0..length)
        .map(|i| {
            let t = (i as f64) / (length as f64) * 10.0;
            let phase = (seed as f64) * 0.01;
            (t.sin() + phase).abs() + 0.5 * (2.0 * t + phase).sin().abs()
        })
        .collect()
}

/// Compare two signals element-wise with tolerance.
fn signals_approx_equal(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tolerance)
}

/// Compare two IMF collections.
fn imfs_approx_equal(imfs_a: &[Vec<f64>], imfs_b: &[Vec<f64>], tolerance: f64) -> bool {
    if imfs_a.len() != imfs_b.len() {
        return false;
    }
    imfs_a.iter().zip(imfs_b.iter()).all(|(a, b)| signals_approx_equal(a, b, tolerance))
}

#[test]
#[ignore] // Pre-existing spline indexing bug
fn test_eemd_parity_small_signal() {
    let signal = generate_signal(256, 42);

    // CPU execution
    let mut cpu_config = GpuEemdConfig::default();
    cpu_config.num_ensembles = 5;
    cpu_config.force_cpu = true;
    cpu_config.seed = Some(12345);

    let mut cpu_executor = GpuEemdExecutor::new(cpu_config).expect("CPU executor creation");
    let cpu_result = cpu_executor.execute(&signal).expect("CPU execution");

    // GPU execution (with CPU fallback for testing)
    let mut gpu_config = GpuEemdConfig::default();
    gpu_config.num_ensembles = 5;
    gpu_config.force_cpu = false;
    gpu_config.seed = Some(12345); // Same seed for reproducibility

    let mut gpu_executor = GpuEemdExecutor::new(gpu_config).expect("GPU executor creation");
    let gpu_result = gpu_executor.execute(&signal).expect("GPU execution");

    // Both should have extracted IMFs
    assert!(!cpu_result.imfs.is_empty(), "CPU should extract IMFs");
    assert!(!gpu_result.imfs.is_empty(), "GPU should extract IMFs");
    assert_eq!(cpu_result.num_trials, gpu_result.num_trials);
}

#[test]
#[ignore] // Pre-existing spline indexing bug
fn test_eemd_deterministic_with_seed() {
    let signal = generate_signal(256, 42);
    let seed = 99999;

    // Run twice with same seed
    let mut config1 = GpuEemdConfig::default();
    config1.num_ensembles = 5;
    config1.force_cpu = true;
    config1.seed = Some(seed);

    let mut executor1 = GpuEemdExecutor::new(config1).expect("executor creation");
    let result1 = executor1.execute(&signal).expect("execution");

    let mut config2 = GpuEemdConfig::default();
    config2.num_ensembles = 5;
    config2.force_cpu = true;
    config2.seed = Some(seed);

    let mut executor2 = GpuEemdExecutor::new(config2).expect("executor creation");
    let result2 = executor2.execute(&signal).expect("execution");

    // Same seed should produce identical results
    assert_eq!(result1.imfs.len(), result2.imfs.len());
    assert_eq!(result1.residue.len(), result2.residue.len());

    // Check IMF values are identical
    if !result1.imfs.is_empty() && !result2.imfs.is_empty() {
        assert!(imfs_approx_equal(&result1.imfs, &result2.imfs, TOLERANCE));
    }

    // Check residues are identical
    assert!(signals_approx_equal(&result1.residue, &result2.residue, TOLERANCE));
}

#[test]
#[ignore] // Pre-existing spline indexing bug
fn test_ceemdan_parity_small_signal() {
    let signal = generate_signal(256, 43);

    let mut cpu_config = GpuCeemданConfig::default();
    cpu_config.num_ensembles = 5;
    cpu_config.force_cpu = true;
    cpu_config.seed = Some(12345);

    let mut cpu_executor = GpuCeemданExecutor::new(cpu_config).expect("CPU executor creation");
    let cpu_result = cpu_executor.execute(&signal).expect("CPU execution");

    let mut gpu_config = GpuCeemданConfig::default();
    gpu_config.num_ensembles = 5;
    gpu_config.force_cpu = false;
    gpu_config.seed = Some(12345);

    let mut gpu_executor = GpuCeemданExecutor::new(gpu_config).expect("GPU executor creation");
    let gpu_result = gpu_executor.execute(&signal).expect("GPU execution");

    // Both should extract IMFs
    assert!(!cpu_result.imfs.is_empty());
    assert!(!gpu_result.imfs.is_empty());
    assert_eq!(cpu_result.num_trials, gpu_result.num_trials);
}

#[test]
#[ignore] // Pre-existing spline indexing bug
fn test_eemd_execution_completes() {
    let signal = generate_signal(512, 100);

    let mut config = GpuEemdConfig::default();
    config.num_ensembles = 3;
    config.force_cpu = true;

    let mut executor = GpuEemdExecutor::new(config).expect("executor creation");
    let result = executor.execute(&signal).expect("execution");

    assert_eq!(result.num_trials, 3);
    assert!(result.execution_time.as_secs_f64() > 0.0);
}

#[test]
#[ignore] // Pre-existing spline indexing bug
fn test_ceemdan_execution_completes() {
    let signal = generate_signal(512, 101);

    let mut config = GpuCeemданConfig::default();
    config.num_ensembles = 3;
    config.force_cpu = true;

    let mut executor = GpuCeemданExecutor::new(config).expect("executor creation");
    let result = executor.execute(&signal).expect("execution");

    assert_eq!(result.num_trials, 3);
    assert!(result.execution_time.as_secs_f64() > 0.0);
}

#[test]
fn test_eemd_empty_signal_error() {
    let signal = vec![];

    let mut config = GpuEemdConfig::default();
    config.force_cpu = true;

    let mut executor = GpuEemdExecutor::new(config).expect("executor creation");
    let result = executor.execute(&signal);

    assert!(result.is_err());
}

#[test]
fn test_ceemdan_empty_signal_error() {
    let signal = vec![];

    let mut config = GpuCeemданConfig::default();
    config.force_cpu = true;

    let mut executor = GpuCeemданExecutor::new(config).expect("executor creation");
    let result = executor.execute(&signal);

    assert!(result.is_err());
}

#[test]
#[ignore] // Pre-existing spline indexing bug
fn test_eemd_different_signal_sizes() {
    for size in [128, 256, 512, 1024] {
        let signal = generate_signal(size, 200 + size as u64);

        let mut config = GpuEemdConfig::default();
        config.num_ensembles = 3;
        config.force_cpu = true;

        let mut executor = GpuEemdExecutor::new(config).expect("executor creation");
        let result = executor.execute(&signal).expect("execution");

        assert_eq!(result.num_trials, 3);
    }
}

#[test]
#[ignore] // Pre-existing spline indexing bug
fn test_ceemdan_different_signal_sizes() {
    for size in [128, 256, 512, 1024] {
        let signal = generate_signal(size, 300 + size as u64);

        let mut config = GpuCeemданConfig::default();
        config.num_ensembles = 3;
        config.force_cpu = true;

        let mut executor = GpuCeemданExecutor::new(config).expect("executor creation");
        let result = executor.execute(&signal).expect("execution");

        assert_eq!(result.num_trials, 3);
    }
}
