fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate gRPC bindings from protobuf
    tonic_build::compile_protos("../../proto/ferromode.proto")?;
    Ok(())
}
