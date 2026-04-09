//! # Ferromode Validator
//! Cross-implementation validation and benchmarking framework

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    pub algorithm: String,
    pub version: String,
    pub imfs: Vec<Vec<f64>>,
    pub residual: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub signal_id: String,
    pub impl1: String,
    pub impl2: String,
    pub correlation: f64,
    pub mse: f64,
}

pub fn correlation(s1: &[f64], s2: &[f64]) -> f64 {
    if s1.len() != s2.len() || s1.is_empty() {
        return f64::NAN;
    }
    let n = s1.len() as f64;
    let mean1: f64 = s1.iter().sum::<f64>() / n;
    let mean2: f64 = s2.iter().sum::<f64>() / n;
    
    let mut num = 0.0_f64;
    let mut sum_sq1 = 0.0_f64;
    let mut sum_sq2 = 0.0_f64;
    
    for (a, b) in s1.iter().zip(s2.iter()) {
        let d1 = a - mean1;
        let d2 = b - mean2;
        num += d1 * d2;
        sum_sq1 += d1 * d1;
        sum_sq2 += d2 * d2;
    }
    
    if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
        return 0.0;
    }
    num / (sum_sq1 * sum_sq2).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation() {
        let s = vec![1.0, 2.0, 3.0];
        assert!((correlation(&s, &s) - 1.0).abs() < 1e-10);
    }
}
