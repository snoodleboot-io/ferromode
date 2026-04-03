use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmdError {
    #[error("empty signal: signal must contain at least one sample")]
    EmptySignal,
    #[error("dimension mismatch: all channels must have the same length")]
    DimensionMismatch,
    #[error("invalid sample rate: sample rate must be positive")]
    InvalidSampleRate,
    #[error("insufficient data: not enough samples for decomposition")]
    InsufficientData,
    #[error("convergence failed: algorithm did not converge after {max_iterations} iterations")]
    ConvergenceFailed { max_iterations: usize },
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("invalid value: signal contains non-finite value (NaN or Inf)")]
    InvalidValue,
}

impl EmdError {
    /// Convert to a Python `ValueError` for pyo3 bindings.
    #[cfg(feature = "python")]
    pub fn to_python_error(&self) -> pyo3::PyErr {
        pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(self.to_string())
    }

    /// Convert to an R `SimpleError` for extendr bindings.
    #[cfg(feature = "r")]
    pub fn to_r_error(&self) -> extendr_api::error::Error {
        extendr_api::error::Error::Other(self.to_string())
    }
}

    /// Convert to an R `SimpleError` for extendr bindings.
    #[cfg(feature = "r")]
    pub fn to_r_error(&self) -> extendr_api::error::Error {
        extendr_api::error::Error::Other(self.to_string())
    }
}

impl From<std::num::ParseFloatError> for EmdError {
    fn from(err: std::num::ParseFloatError) -> Self {
        EmdError::InvalidConfig(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emd_error_display_messages() {
        assert_eq!(
            EmdError::EmptySignal.to_string(),
            "empty signal: signal must contain at least one sample"
        );
        assert_eq!(
            EmdError::DimensionMismatch.to_string(),
            "dimension mismatch: all channels must have the same length"
        );
        assert_eq!(
            EmdError::InvalidSampleRate.to_string(),
            "invalid sample rate: sample rate must be positive"
        );
        assert_eq!(
            EmdError::InsufficientData.to_string(),
            "insufficient data: not enough samples for decomposition"
        );
        assert_eq!(
            EmdError::ConvergenceFailed { max_iterations: 100 }.to_string(),
            "convergence failed: algorithm did not converge after 100 iterations"
        );
        assert_eq!(
            EmdError::InvalidConfig("bad value".to_string()).to_string(),
            "invalid config: bad value"
        );
        assert_eq!(
            EmdError::InvalidValue.to_string(),
            "invalid value: signal contains non-finite value (NaN or Inf)"
        );
    }

    #[test]
    fn test_emd_error_can_be_used_as_std_error() {
        fn takes_std_error(_err: &dyn std::error::Error) {}
        takes_std_error(&EmdError::EmptySignal);
        takes_std_error(&EmdError::InvalidConfig("test".to_string()));
    }

    #[test]
    fn test_from_parse_float_error() {
        let result: Result<f64, EmdError> = "not_a_number".parse().map_err(EmdError::from);
        assert!(result.is_err());
        match result.unwrap_err() {
            EmdError::InvalidConfig(msg) => assert!(!msg.is_empty()),
            other => panic!("expected InvalidConfig, got {:?}", other),
        }
    }
}
