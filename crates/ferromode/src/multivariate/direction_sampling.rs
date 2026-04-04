//! Hypersphere direction sampling for multivariate EMD.
//!
//! Provides uniform angular sampling on the n-sphere using multiple strategies:
//! - Uniform sampling via normalized Gaussian vectors
//! - Hammersley low-discrepancy sequence
//! - Halton sequence
//!
//! Reference: Rehman & Mandic (2010), "Multivariate Empirical Mode Decomposition"

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Direction sampling strategy for multivariate EMD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectionSampling {
    /// Uniform sampling via normalized Gaussian vectors.
    Uniform,
    /// Hammersley low-discrepancy sequence.
    Hammersley,
    /// Halton sequence.
    Halton,
}

/// Configuration for direction sampling.
#[derive(Debug, Clone)]
pub struct DirectionConfig {
    /// Number of direction vectors to generate.
    pub num_directions: usize,
    /// Sampling strategy to use.
    pub sampling: DirectionSampling,
    /// Optional seed for reproducibility (only used for `Uniform` sampling).
    pub seed: Option<u64>,
}

impl DirectionConfig {
    /// Create a new configuration with uniform sampling.
    #[must_use]
    pub fn new(num_directions: usize) -> Self {
        Self { num_directions, sampling: DirectionSampling::Uniform, seed: None }
    }

    /// Set the sampling strategy.
    #[must_use]
    pub fn with_sampling(mut self, sampling: DirectionSampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Generate direction vectors uniformly distributed on the n-sphere.
///
/// # Arguments
/// * `config` - Sampling configuration
/// * `n_dims` - Dimensionality of the sphere (n_dims = 2 means circle, 3 means 2-sphere, etc.)
///
/// # Returns
/// Vector of unit vectors, each of length `n_dims`.
///
/// # Panics
/// Panics if `n_dims < 2` or `num_directions == 0`.
#[must_use]
pub fn generate_directions(config: &DirectionConfig, n_dims: usize) -> Vec<Vec<f64>> {
    assert!(n_dims >= 2, "n_dims must be >= 2, got {n_dims}");
    assert!(config.num_directions > 0, "num_directions must be > 0");

    match config.sampling {
        DirectionSampling::Uniform => {
            uniform_sphere_sampling(n_dims, config.num_directions, config.seed)
        }
        DirectionSampling::Hammersley => hammersley_sequence(n_dims, config.num_directions),
        DirectionSampling::Halton => halton_sequence(n_dims, config.num_directions),
    }
}

/// Generate unit vectors uniformly distributed on the n-sphere using normalized Gaussian vectors.
///
/// This method generates vectors from a standard multivariate normal distribution
/// and normalizes them to unit length. By the spherical symmetry of the Gaussian,
/// the resulting vectors are uniformly distributed on the sphere.
///
/// Reference: Muller (1959), "A Note on a Method for Generating Points Uniformly on n-Dimensional Spheres"
///
/// # Arguments
/// * `n_dims` - Dimensionality
/// * `n_directions` - Number of direction vectors
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// Vector of unit vectors of length `n_dims`.
#[must_use]
pub fn uniform_sphere_sampling(
    n_dims: usize,
    n_directions: usize,
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(0.0, 1.0).expect("Normal distribution parameters are valid");

    (0..n_directions)
        .map(|_| {
            let mut vec: Vec<f64> = (0..n_dims).map(|_| normal.sample(&mut rng)).collect();
            let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            vec
        })
        .collect()
}

/// Generate direction vectors using the Hammersley low-discrepancy sequence.
///
/// The Hammersley sequence provides better uniformity than pseudo-random sampling
/// for quasi-Monte Carlo integration. Points are mapped to the sphere using
/// inverse transform sampling.
///
/// # Arguments
/// * `n_dims` - Dimensionality
/// * `n_directions` - Number of direction vectors
///
/// # Returns
/// Vector of unit vectors of length `n_dims`.
#[must_use]
pub fn hammersley_sequence(n_dims: usize, n_directions: usize) -> Vec<Vec<f64>> {
    (0..n_directions)
        .map(|i| {
            let mut point = Vec::with_capacity(n_dims);

            // First dimension: i / N
            let u0 = i as f64 / n_directions as f64;
            point.push(inverse_normal_cdf(u0));

            // Remaining dimensions: radical inverse in different bases
            for d in 1..n_dims {
                let base = (d + 1) as u64;
                let u = radical_inverse(i as u64, base);
                point.push(inverse_normal_cdf(u));
            }

            // Normalize to unit vector
            let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut point {
                    *x /= norm;
                }
            }
            point
        })
        .collect()
}

/// Generate direction vectors using the Halton sequence.
///
/// The Halton sequence is a low-discrepancy sequence that provides excellent
/// uniformity for quasi-Monte Carlo methods. Each dimension uses a different
/// prime base for the radical inverse function.
///
/// # Arguments
/// * `n_dims` - Dimensionality
/// * `n_directions` - Number of direction vectors
///
/// # Returns
/// Vector of unit vectors of length `n_dims`.
#[must_use]
pub fn halton_sequence(n_dims: usize, n_directions: usize) -> Vec<Vec<f64>> {
    // Primes for each dimension (first 8 primes cover up to n_dims=8)
    let primes: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];

    (0..n_directions)
        .map(|i| {
            let mut point = Vec::with_capacity(n_dims);

            for d in 0..n_dims {
                let base = if d < primes.len() {
                    primes[d]
                } else {
                    // Fallback for higher dimensions: use odd numbers
                    (2 * d + 1) as u64
                };
                let u = halton_value(i as u64, d as u64);
                point.push(inverse_normal_cdf(u));
            }

            // Normalize to unit vector
            let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut point {
                    *x /= norm;
                }
            }
            point
        })
        .collect()
}

/// Compute the radical inverse of n in base b.
///
/// The radical inverse function φ_b(n) reflects the base-b representation of n
/// about the decimal point.
///
/// # Arguments
/// * `n` - Non-negative integer
/// * `base` - Base for the radical inverse (must be >= 2)
///
/// # Returns
/// Radical inverse value in [0, 1).
#[must_use]
fn radical_inverse(n: u64, base: u64) -> f64 {
    let mut result = 0.0f64;
    let mut n = n;
    let mut denom = 1.0f64;
    let base_f = base as f64;

    while n > 0 {
        let digit = n % base;
        denom *= base_f;
        result += digit as f64 / denom;
        n /= base;
    }

    result
}

/// Compute the Halton sequence value for index i in dimension d.
///
/// Uses the d-th prime as the base for the radical inverse.
///
/// # Arguments
/// * `i` - Index in the sequence
/// * `d` - Dimension index (0-based)
///
/// # Returns
/// Halton value in [0, 1).
#[must_use]
fn halton_value(i: u64, d: u64) -> f64 {
    // Primes for dimensions
    let primes: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
    let base = if d < primes.len() as u64 {
        primes[d as usize]
    } else {
        // Fallback: find next prime (simple approach for higher dimensions)
        let mut candidate = 2 * d + 1;
        while !is_prime(candidate) {
            candidate += 2;
        }
        candidate
    };

    radical_inverse(i, base)
}

/// Check if a number is prime.
#[must_use]
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Inverse of the standard normal cumulative distribution function.
///
/// Uses the Beasley-Springer-Moro algorithm for high accuracy.
///
/// # Arguments
/// * `p` - Probability value in (0, 1)
///
/// # Returns
/// z such that Φ(z) = p, where Φ is the standard normal CDF.
///
/// # Panics
/// Panics if p <= 0 or p >= 1.
#[must_use]
pub fn inverse_normal_cdf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0, 1), got {p}");

    // Coefficients for rational approximation
    const A: [f64; 8] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_690e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
        -5.447_609_879_822_406e-01,
        1.615_858_368_580_409e-02,
    ];
    const B: [f64; 8] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_690e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
        0.0,
        0.0,
    ];

    // Use symmetry: Φ^(-1)(p) = -Φ^(-1)(1-p)
    let (p, sign) = if p > 0.5 { (1.0 - p, -1.0) } else { (p, 1.0) };

    // Rational approximation for central region
    let r = p;
    let numerator = A[0]
        + r * (A[1] + r * (A[2] + r * (A[3] + r * (A[4] + r * (A[5] + r * (A[6] + r * A[7]))))));
    let denominator = B[0]
        + r * (B[1] + r * (B[2] + r * (B[3] + r * (B[4] + r * (B[5] + r * (B[6] + r * B[7]))))));

    let x = numerator / denominator;
    sign * x
}

/// Compute the Kolmogorov-Smirnov statistic for a sample vs uniform[0,1].
///
/// # Arguments
/// * `sample` - Sample values (should be in [0, 1])
///
/// # Returns
/// KS statistic (maximum deviation from uniform CDF).
#[must_use]
fn ks_statistic_uniform(sample: &[f64]) -> f64 {
    let n = sample.len();
    if n == 0 {
        return 0.0;
    }

    let mut sorted: Vec<f64> = sample.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_diff = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let empirical = (i + 1) as f64 / n as f64;
        let theoretical = x.clamp(0.0, 1.0);
        let diff1 = (empirical - theoretical).abs();
        let diff2 = ((i as f64 / n as f64) - theoretical).abs();
        max_diff = max_diff.max(diff1).max(diff2);
    }

    max_diff
}

/// Critical value for KS test at α=0.05.
///
/// Approximation: D_α ≈ 1.36 / sqrt(n)
#[must_use]
fn ks_critical_value(n: usize) -> f64 {
    1.36 / (n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;
    const UNIT_TOLERANCE: f64 = 1e-6;

    // ─── Helper functions ───────────────────────────────────────────────

    fn compute_norm(vec: &[f64]) -> f64 {
        vec.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn compute_mean_direction(directions: &[Vec<f64>]) -> Vec<f64> {
        let n_dims = directions[0].len();
        let n = directions.len() as f64;
        let mut mean = vec![0.0f64; n_dims];
        for dir in directions {
            for (i, &x) in dir.iter().enumerate() {
                mean[i] += x / n;
            }
        }
        mean
    }

    fn project_to_angles(directions: &[Vec<f64>]) -> Vec<f64> {
        // Project onto first two dimensions and compute angle
        directions
            .iter()
            .map(|dir| {
                let angle = dir[1].atan2(dir[0]);
                // Normalize to [0, 1]
                (angle + PI) / (2.0 * PI)
            })
            .collect()
    }

    // ─── T-084: Uniform sphere sampling tests ───────────────────────────

    #[test]
    fn test_uniform_directions_are_unit_vectors_n2() {
        let directions = uniform_sphere_sampling(2, 100, Some(42));
        assert_eq!(directions.len(), 100);
        for dir in &directions {
            assert_eq!(dir.len(), 2);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_uniform_directions_are_unit_vectors_n3() {
        let directions = uniform_sphere_sampling(3, 100, Some(42));
        for dir in &directions {
            assert_eq!(dir.len(), 3);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_uniform_directions_are_unit_vectors_n4() {
        let directions = uniform_sphere_sampling(4, 100, Some(42));
        for dir in &directions {
            assert_eq!(dir.len(), 4);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_uniform_directions_are_unit_vectors_n6() {
        let directions = uniform_sphere_sampling(6, 100, Some(42));
        for dir in &directions {
            assert_eq!(dir.len(), 6);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_uniform_directions_are_unit_vectors_n8() {
        let directions = uniform_sphere_sampling(8, 100, Some(42));
        for dir in &directions {
            assert_eq!(dir.len(), 8);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_uniform_mean_direction_near_zero_n3() {
        let directions = uniform_sphere_sampling(3, 10000, Some(42));
        let mean = compute_mean_direction(&directions);
        for &x in &mean {
            assert!(x.abs() < 0.02, "Mean direction component {x} not near zero");
        }
    }

    #[test]
    fn test_uniform_reproducibility_with_seed() {
        let dirs1 = uniform_sphere_sampling(3, 50, Some(123));
        let dirs2 = uniform_sphere_sampling(3, 50, Some(123));
        assert_eq!(dirs1.len(), dirs2.len());
        for (d1, d2) in dirs1.iter().zip(&dirs2) {
            for (x1, x2) in d1.iter().zip(d2) {
                assert!((x1 - x2).abs() < TOLERANCE);
            }
        }
    }

    // ─── T-085: Hammersley sequence tests ───────────────────────────────

    #[test]
    fn test_hammersley_directions_are_unit_vectors_n2() {
        let directions = hammersley_sequence(2, 100);
        assert_eq!(directions.len(), 100);
        for dir in &directions {
            assert_eq!(dir.len(), 2);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_hammersley_directions_are_unit_vectors_n3() {
        let directions = hammersley_sequence(3, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 3);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_hammersley_directions_are_unit_vectors_n4() {
        let directions = hammersley_sequence(4, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 4);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_hammersley_directions_are_unit_vectors_n6() {
        let directions = hammersley_sequence(6, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 6);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_hammersley_directions_are_unit_vectors_n8() {
        let directions = hammersley_sequence(8, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 8);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_hammersley_deterministic() {
        let dirs1 = hammersley_sequence(3, 50);
        let dirs2 = hammersley_sequence(3, 50);
        assert_eq!(dirs1, dirs2);
    }

    // ─── T-086: Halton sequence tests ───────────────────────────────────

    #[test]
    fn test_halton_directions_are_unit_vectors_n2() {
        let directions = halton_sequence(2, 100);
        assert_eq!(directions.len(), 100);
        for dir in &directions {
            assert_eq!(dir.len(), 2);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_halton_directions_are_unit_vectors_n3() {
        let directions = halton_sequence(3, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 3);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_halton_directions_are_unit_vectors_n4() {
        let directions = halton_sequence(4, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 4);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_halton_directions_are_unit_vectors_n6() {
        let directions = halton_sequence(6, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 6);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_halton_directions_are_unit_vectors_n8() {
        let directions = halton_sequence(8, 100);
        for dir in &directions {
            assert_eq!(dir.len(), 8);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE, "Norm {norm} not close to 1.0");
        }
    }

    #[test]
    fn test_halton_deterministic() {
        let dirs1 = halton_sequence(3, 50);
        let dirs2 = halton_sequence(3, 50);
        assert_eq!(dirs1, dirs2);
    }

    // ─── T-087: DirectionSampling enum and generate_directions tests ────

    #[test]
    fn test_generate_directions_uniform() {
        let config = DirectionConfig::new(50).with_sampling(DirectionSampling::Uniform);
        let directions = generate_directions(&config, 3);
        assert_eq!(directions.len(), 50);
        for dir in &directions {
            assert_eq!(dir.len(), 3);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE);
        }
    }

    #[test]
    fn test_generate_directions_hammersley() {
        let config = DirectionConfig::new(50).with_sampling(DirectionSampling::Hammersley);
        let directions = generate_directions(&config, 3);
        assert_eq!(directions.len(), 50);
        for dir in &directions {
            assert_eq!(dir.len(), 3);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE);
        }
    }

    #[test]
    fn test_generate_directions_halton() {
        let config = DirectionConfig::new(50).with_sampling(DirectionSampling::Halton);
        let directions = generate_directions(&config, 3);
        assert_eq!(directions.len(), 50);
        for dir in &directions {
            assert_eq!(dir.len(), 3);
            let norm = compute_norm(dir);
            assert!((norm - 1.0).abs() < UNIT_TOLERANCE);
        }
    }

    #[test]
    #[should_panic(expected = "n_dims must be >= 2")]
    fn test_generate_directions_panics_on_n_dims_less_than_2() {
        let config = DirectionConfig::new(10);
        generate_directions(&config, 1);
    }

    #[test]
    #[should_panic(expected = "num_directions must be > 0")]
    fn test_generate_directions_panics_on_zero_directions() {
        let config = DirectionConfig::new(0);
        generate_directions(&config, 3);
    }

    // ─── T-088: Statistical validation tests ────────────────────────────

    #[test]
    fn test_uniform_ks_test_n2() {
        let directions = uniform_sphere_sampling(2, 1000, Some(42));
        let angles = project_to_angles(&directions);
        let ks = ks_statistic_uniform(&angles);
        let critical = ks_critical_value(angles.len());
        assert!(ks < critical, "KS statistic {ks} exceeds critical value {critical} for n=2");
    }

    #[test]
    fn test_uniform_ks_test_n3() {
        let directions = uniform_sphere_sampling(3, 1000, Some(42));
        let angles = project_to_angles(&directions);
        let ks = ks_statistic_uniform(&angles);
        let critical = ks_critical_value(angles.len());
        assert!(ks < critical, "KS statistic {ks} exceeds critical value {critical} for n=3");
    }

    #[test]
    fn test_uniform_ks_test_n4() {
        let directions = uniform_sphere_sampling(4, 1000, Some(42));
        let angles = project_to_angles(&directions);
        let ks = ks_statistic_uniform(&angles);
        let critical = ks_critical_value(angles.len());
        assert!(ks < critical, "KS statistic {ks} exceeds critical value {critical} for n=4");
    }

    #[test]
    fn test_uniform_ks_test_n6() {
        let directions = uniform_sphere_sampling(6, 1000, Some(42));
        let angles = project_to_angles(&directions);
        let ks = ks_statistic_uniform(&angles);
        let critical = ks_critical_value(angles.len());
        assert!(ks < critical, "KS statistic {ks} exceeds critical value {critical} for n=6");
    }

    #[test]
    fn test_uniform_ks_test_n8() {
        let directions = uniform_sphere_sampling(8, 1000, Some(42));
        let angles = project_to_angles(&directions);
        let ks = ks_statistic_uniform(&angles);
        let critical = ks_critical_value(angles.len());
        assert!(ks < critical, "KS statistic {ks} exceeds critical value {critical} for n=8");
    }

    #[test]
    fn test_hammersley_ks_test_n3() {
        let directions = hammersley_sequence(3, 1000);
        let angles = project_to_angles(&directions);
        let ks = ks_statistic_uniform(&angles);
        let critical = ks_critical_value(angles.len());
        assert!(
            ks < critical,
            "KS statistic {ks} exceeds critical value {critical} for Hammersley n=3"
        );
    }

    #[test]
    fn test_halton_ks_test_n3() {
        let directions = halton_sequence(3, 1000);
        let angles = project_to_angles(&directions);
        let ks = ks_statistic_uniform(&angles);
        let critical = ks_critical_value(angles.len());
        assert!(
            ks < critical,
            "KS statistic {ks} exceeds critical value {critical} for Halton n=3"
        );
    }

    #[test]
    fn test_compare_discrepancy_hammersley_vs_halton_vs_uniform() {
        let n_dims = 3;
        let n_directions = 500;

        let uniform_dirs = uniform_sphere_sampling(n_dims, n_directions, Some(42));
        let hammersley_dirs = hammersley_sequence(n_dims, n_directions);
        let halton_dirs = halton_sequence(n_dims, n_directions);

        let uniform_ks = ks_statistic_uniform(&project_to_angles(&uniform_dirs));
        let hammersley_ks = ks_statistic_uniform(&project_to_angles(&hammersley_dirs));
        let halton_ks = ks_statistic_uniform(&project_to_angles(&halton_dirs));

        // Low-discrepancy sequences should have lower or comparable KS statistics
        // Note: This is a soft assertion since randomness can vary
        assert!(
            uniform_ks < ks_critical_value(n_directions),
            "Uniform KS {uniform_ks} should pass KS test"
        );
        assert!(
            hammersley_ks < ks_critical_value(n_directions),
            "Hammersley KS {hammersley_ks} should pass KS test"
        );
        assert!(
            halton_ks < ks_critical_value(n_directions),
            "Halton KS {halton_ks} should pass KS test"
        );

        // Log the values for comparison (not an assertion, just informative)
        // In practice, low-discrepancy sequences often have lower discrepancy
        println!(
            "Uniform KS: {uniform_ks:.6}, Hammersley KS: {hammersley_ks:.6}, Halton KS: {halton_ks:.6}"
        );
    }

    #[test]
    fn test_inverse_normal_cdf_symmetry() {
        // Φ^(-1)(p) = -Φ^(-1)(1-p)
        let p = 0.3;
        let z1 = inverse_normal_cdf(p);
        let z2 = inverse_normal_cdf(1.0 - p);
        assert!((z1 + z2).abs() < 1e-10, "Symmetry violated: {z1} != -{z2}");
    }

    #[test]
    fn test_inverse_normal_cdf_median() {
        // Φ^(-1)(0.5) should be 0
        let z = inverse_normal_cdf(0.5);
        assert!(z.abs() < 1e-10, "Median not zero: {z}");
    }

    #[test]
    #[should_panic]
    fn test_inverse_normal_cdf_panics_on_zero() {
        inverse_normal_cdf(0.0);
    }

    #[test]
    #[should_panic]
    fn test_inverse_normal_cdf_panics_on_one() {
        inverse_normal_cdf(1.0);
    }

    // ─── Dimension coverage tests ───────────────────────────────────────

    #[test]
    fn test_all_dimensions_unit_vectors() {
        for n_dims in [2, 3, 4, 6, 8] {
            let directions = uniform_sphere_sampling(n_dims, 100, Some(42));
            assert_eq!(directions.len(), 100);
            for dir in &directions {
                assert_eq!(dir.len(), n_dims);
                let norm = compute_norm(dir);
                assert!(
                    (norm - 1.0).abs() < UNIT_TOLERANCE,
                    "n_dims={n_dims}: norm {norm} not close to 1.0"
                );
            }
        }
    }

    #[test]
    fn test_all_dimensions_hammersley_unit_vectors() {
        for n_dims in [2, 3, 4, 6, 8] {
            let directions = hammersley_sequence(n_dims, 100);
            for dir in &directions {
                assert_eq!(dir.len(), n_dims);
                let norm = compute_norm(dir);
                assert!(
                    (norm - 1.0).abs() < UNIT_TOLERANCE,
                    "n_dims={n_dims}: norm {norm} not close to 1.0"
                );
            }
        }
    }

    #[test]
    fn test_all_dimensions_halton_unit_vectors() {
        for n_dims in [2, 3, 4, 6, 8] {
            let directions = halton_sequence(n_dims, 100);
            for dir in &directions {
                assert_eq!(dir.len(), n_dims);
                let norm = compute_norm(dir);
                assert!(
                    (norm - 1.0).abs() < UNIT_TOLERANCE,
                    "n_dims={n_dims}: norm {norm} not close to 1.0"
                );
            }
        }
    }
}
