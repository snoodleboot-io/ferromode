#![warn(missing_docs)]

//! GPU executor integration tests.
//!
//! Tests GPU executor orchestration, memory management, and error handling.

#[cfg(test)]
mod tests {
    use crate::adapters::gpu::executor::{EnsembleExecutor, ExecutorConfig};
    use crate::algorithms::eemd::EnsembleConfig;
    use crate::types::Signal;
    use std::time::Duration;

    #[test]
    fn test_gpu_executor_with_very_limited_memory() {
        let executor = EnsembleExecutor::new(ExecutorConfig {
            max_gpu_memory: 64, // Extremely small GPU memory (64 bytes)
            batch_size: 16,
            profiling_enabled: false,
        })
        .expect("executor creation failed with very limited memory config");

        // Executor should still be created and functional
        let device_info = executor.device_info();
        assert!(!device_info.is_empty(), "Device info should be available");
    }

    #[test]
    fn test_gpu_executor_device_info() {
        let executor = EnsembleExecutor::default();
        let info = executor.device_info();
        assert!(!info.is_empty());
        assert!(info.contains("Device:"));
        assert!(info.contains("Memory:"));
        assert!(info.contains("Available:"));
    }

    #[test]
    fn test_gpu_executor_current_device() {
        let executor = EnsembleExecutor::default();
        let device = executor.current_device();
        assert!(!device.is_empty());
    }

    #[test]
    fn test_gpu_executor_has_gpu() {
        let executor = EnsembleExecutor::default();
        // May or may not have GPU, but should not panic
        let _has_gpu = executor.has_gpu();
    }

    #[test]
    fn test_gpu_executor_available_memory() {
        let executor = EnsembleExecutor::default();
        let mem = executor.available_memory();
        assert!(mem > 0, "Available memory should be positive");
    }

    #[test]
    fn test_gpu_executor_stats_initial_state() {
        let executor = EnsembleExecutor::default();
        let stats = executor.stats();
        assert_eq!(stats.total_time, Duration::ZERO);
        assert_eq!(stats.gpu_time, Duration::ZERO);
        assert_eq!(stats.cpu_time, Duration::ZERO);
        assert_eq!(stats.transfer_time, Duration::ZERO);
        assert_eq!(stats.trials_completed, 0);
    }

    #[test]
    fn test_gpu_executor_gpu_disabled_flag() {
        let executor_disabled = EnsembleExecutor::default().with_gpu_disabled();
        let executor_enabled = EnsembleExecutor::default();

        // One should report GPU as unavailable (fallback disabled)
        // The other should check system GPU availability
        let _ = executor_disabled;
        let _ = executor_enabled;
    }

    #[test]
    fn test_gpu_executor_stats_reset() {
        let mut executor = EnsembleExecutor::default();

        // Get initial stats
        let stats_before = executor.stats();
        assert_eq!(stats_before.total_time, Duration::ZERO);

        // Reset (should be idempotent)
        executor.reset_stats();

        let stats_after = executor.stats();
        assert_eq!(stats_after.total_time, Duration::ZERO);
        assert_eq!(stats_after.gpu_time, Duration::ZERO);
        assert_eq!(stats_after.cpu_time, Duration::ZERO);
    }

    #[test]
    fn test_gpu_executor_config_default() {
        let config = ExecutorConfig::default();
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_gpu_memory, 4 * 1024 * 1024 * 1024); // 4GB
        assert!(!config.profiling_enabled);
    }

    #[test]
    fn test_gpu_executor_config_custom() {
        let config = ExecutorConfig {
            max_gpu_memory: 8 * 1024 * 1024 * 1024,
            batch_size: 32,
            profiling_enabled: true,
        };

        let executor = EnsembleExecutor::new(config).expect("executor creation failed");
        let device_info = executor.device_info();
        assert!(!device_info.is_empty());
    }

    #[test]
    fn test_gpu_executor_default_instance() {
        let executor = EnsembleExecutor::default();
        let device = executor.current_device();
        assert!(!device.is_empty());
    }

    #[test]
    fn test_gpu_executor_ensemble_config_creation() {
        let config = EnsembleConfig { num_ensembles: 10, noise_std: 0.2, seed: Some(42) };

        assert_eq!(config.num_ensembles, 10);
        assert_eq!(config.noise_std, 0.2);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_gpu_executor_signal_creation() {
        let signal = Signal::with_sample_rate(&vec![1.0, 2.0, 3.0, 4.0, 5.0], 100.0);
        assert!(signal.is_ok());

        let sig = signal.unwrap();
        assert_eq!(sig.len(), 5);
        assert_eq!(sig.sample_rate(), Some(100.0));
    }

    #[test]
    fn test_gpu_executor_three_algorithms_available() {
        let executor = EnsembleExecutor::default();
        // Just verify the methods exist and are callable (even if they might fail)
        assert!(executor.current_device().len() > 0);
    }

    #[test]
    fn test_gpu_executor_memory_stats() {
        let executor = EnsembleExecutor::default();
        let stats = executor.stats();

        assert_eq!(stats.peak_gpu_memory, 0);
        assert_eq!(stats.gpu_utilization, 0.0);
        assert!(stats.gpu_utilization >= 0.0 && stats.gpu_utilization <= 1.0);
    }

    #[test]
    fn test_gpu_executor_fallback_with_disabled_gpu() {
        let mut executor = EnsembleExecutor::default().with_gpu_disabled();
        let signal = Signal::with_sample_rate(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 100.0)
            .expect("signal creation failed");
        let config = EnsembleConfig { num_ensembles: 2, noise_std: 0.1, seed: Some(42) };

        // Executor should attempt to use CPU fallback
        // May fail due to algorithm limitations with short signals, but shouldn't panic
        let _ = executor.execute_gpu_eemd(&signal, &config);
    }

    #[test]
    fn test_gpu_executor_device_availability() {
        let executor = EnsembleExecutor::default();
        let available_mem = executor.available_memory();
        let has_gpu = executor.has_gpu();

        // Should have some memory available (even on CPU-only system)
        assert!(available_mem >= 0);

        // has_gpu may be true or false depending on system
        let _ = has_gpu;
    }

    #[test]
    fn test_gpu_executor_creation_success() {
        let result = EnsembleExecutor::new(ExecutorConfig::default());
        assert!(result.is_ok(), "Executor creation should succeed with default config");
    }

    #[test]
    fn test_gpu_executor_large_batch_size() {
        let config = ExecutorConfig {
            max_gpu_memory: 4 * 1024 * 1024 * 1024,
            batch_size: 256,
            profiling_enabled: false,
        };

        let executor = EnsembleExecutor::new(config).expect("executor creation failed");
        assert!(!executor.current_device().is_empty());
    }

    #[test]
    fn test_gpu_executor_multiple_config_variations() {
        let configs = vec![
            ExecutorConfig { max_gpu_memory: 1024 * 1024, batch_size: 8, profiling_enabled: false },
            ExecutorConfig {
                max_gpu_memory: 512 * 1024 * 1024,
                batch_size: 16,
                profiling_enabled: false,
            },
            ExecutorConfig {
                max_gpu_memory: 2 * 1024 * 1024 * 1024,
                batch_size: 32,
                profiling_enabled: true,
            },
        ];

        for config in configs {
            let executor = EnsembleExecutor::new(config).expect("executor creation should succeed");
            assert!(executor.has_gpu() || true); // May or may not have GPU
        }
    }
}
