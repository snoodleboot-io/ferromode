#![warn(missing_docs)]

//! Ferromode gRPC microservice for distributed EMD decomposition.
//!
//! Provides a tonic-based gRPC server exposing EMD decomposition
//! as a distributed service.

pub mod service;

pub use service::{pb, EmdServiceImpl};

// Re-export for convenience
pub use service::EmdServiceImpl as EmdService;
