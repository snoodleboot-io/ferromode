use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::types::Signal;
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

/// Auto-generated protobuf types for the ferromode.v1 service contract.
pub mod pb {
    // Proto-generated code carries no doc comments.
    #![allow(missing_docs)]
    tonic::include_proto!("ferromode.v1");
}

use pb::{
    emd_service_server::EmdService, DecomposeRequest, DecomposeResponse, HealthRequest,
    HealthResponse, ImfFrame, Signal as PbSignal, SignalChunk,
};

/// EMD gRPC service implementation
#[derive(Debug, Default, Clone)]
pub struct EmdServiceImpl;

#[tonic::async_trait]
impl EmdService for EmdServiceImpl {
    async fn decompose(
        &self,
        request: Request<DecomposeRequest>,
    ) -> Result<Response<DecomposeResponse>, Status> {
        let req = request.into_inner();

        let pb_signal = req.signal.ok_or_else(|| Status::invalid_argument("Missing signal"))?;
        if pb_signal.values.is_empty() {
            return Err(Status::invalid_argument("Signal cannot be empty"));
        }

        let signal = Signal::from_slice(&pb_signal.values)
            .map_err(|e| Status::internal(format!("Invalid signal: {e:?}")))?;

        let mut config = EmdConfig::default();
        if let Some(c) = &req.config {
            config.max_imfs = c.max_imfs as usize;
        }

        match emd(signal.values(), &config) {
            Ok(result) => {
                let imfs = result
                    .imfs
                    .imfs
                    .iter()
                    .map(|imf| PbSignal {
                        values: imf.clone(),
                        sample_rate: signal.sample_rate().unwrap_or(1.0),
                    })
                    .collect();

                let response = DecomposeResponse {
                    imfs,
                    total_sift_iterations: result.n_siftings as i32,
                    error: String::new(),
                    success: true,
                };

                Ok(Response::new(response))
            }
            Err(e) => {
                let response = DecomposeResponse {
                    imfs: vec![],
                    total_sift_iterations: 0,
                    error: format!("Decomposition failed: {e:?}"),
                    success: false,
                };
                Ok(Response::new(response))
            }
        }
    }

    type DecomposeStreamStream =
        Pin<Box<dyn futures::Stream<Item = Result<ImfFrame, Status>> + Send>>;

    async fn decompose_stream(
        &self,
        request: Request<tonic::Streaming<SignalChunk>>,
    ) -> Result<Response<Self::DecomposeStreamStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        tokio::spawn(async move {
            let mut signal_data = Vec::new();
            let mut imf_index = 0;

            while let Some(chunk_result) = stream.message().await.transpose() {
                match chunk_result {
                    Ok(chunk) => {
                        let values = Self::bytes_to_f64_vec(&chunk.data);
                        signal_data.extend(values);

                        if chunk.is_final {
                            if let Ok(signal) = Signal::from_slice(&signal_data) {
                                let config = EmdConfig::default();

                                if let Ok(result) = emd(signal.values(), &config) {
                                    for imf in &result.imfs.imfs {
                                        imf_index += 1;
                                        let frame = ImfFrame {
                                            data: Self::f64_vec_to_bytes(imf),
                                            imf_index,
                                            iterations: 1,
                                        };
                                        let _ = tx.send(Ok(frame)).await;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(format!("Stream error: {e}")))).await;
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { status: pb::health_response::Status::Serving as i32 }))
    }
}

impl EmdServiceImpl {
    fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
        bytes
            .chunks_exact(8)
            .filter_map(|chunk| {
                if let Ok(arr) = <[u8; 8]>::try_from(chunk) {
                    Some(f64::from_le_bytes(arr))
                } else {
                    None
                }
            })
            .collect()
    }

    fn f64_vec_to_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|&v| v.to_le_bytes().to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_f64_conversion() {
        let values = vec![1.0, 2.0, 3.0];
        let bytes = EmdServiceImpl::f64_vec_to_bytes(&values);
        let recovered = EmdServiceImpl::bytes_to_f64_vec(&bytes);
        assert_eq!(values, recovered);
    }

    #[tokio::test]
    async fn test_decompose_simple_signal() {
        let service = EmdServiceImpl::default();

        // Use an oscillatory signal so EMD can extract at least one IMF.
        // A monotone ramp has no local extrema and produces zero IMFs.
        let values: Vec<f64> = (0..64)
            .map(|i| (std::f64::consts::PI * 2.0 * i as f64 / 16.0).sin())
            .collect();
        let signal = PbSignal { values, sample_rate: 1.0 };

        let request = Request::new(DecomposeRequest { signal: Some(signal), config: None });

        let response = service.decompose(request).await;
        assert!(response.is_ok());

        let response = response.unwrap().into_inner();
        assert!(response.success);
        assert!(!response.imfs.is_empty());
    }

    #[tokio::test]
    async fn test_decompose_empty_signal() {
        let service = EmdServiceImpl::default();

        let signal = PbSignal { values: vec![], sample_rate: 1.0 };

        let request = Request::new(DecomposeRequest { signal: Some(signal), config: None });

        let response = service.decompose(request).await;
        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_health_check() {
        let service = EmdServiceImpl::default();

        let request =
            Request::new(HealthRequest { service: "ferromode.v1.EmdService".to_string() });

        let response = service.health(request).await;
        assert!(response.is_ok());

        let response = response.unwrap().into_inner();
        assert_eq!(response.status, pb::health_response::Status::Serving as i32);
    }
}
