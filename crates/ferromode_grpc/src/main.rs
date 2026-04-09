use ferromode_grpc::pb::emd_service_server::EmdServiceServer;
use ferromode_grpc::EmdServiceImpl;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse listen address from environment or use default
    let addr_str = std::env::var("EMD_LISTEN_ADDR").unwrap_or_else(|_| "[::1]:50051".to_string());
    let addr: SocketAddr = addr_str.parse()?;

    println!("🚀 Ferromode EMD gRPC service starting on {}", addr);

    // Create the service
    let emd_service = EmdServiceImpl::default();

    // Build and start the server
    Server::builder().add_service(EmdServiceServer::new(emd_service)).serve(addr).await?;

    Ok(())
}
