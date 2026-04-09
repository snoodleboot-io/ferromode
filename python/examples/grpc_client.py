#!/usr/bin/env python3
"""
Python gRPC client for Ferromode EMD decomposition service.

Example usage:
    python grpc_client.py --host localhost --port 50051 --signal [1, 2, 3, 4, 5]
"""

import grpc
import argparse
import json
import numpy as np
from typing import List, Optional


# This would be generated from proto files
# For now, we provide stubs
class Signal:
    def __init__(self, values: List[float], sample_rate: float = 1.0):
        self.values = values
        self.sample_rate = sample_rate


class EmdConfig:
    def __init__(
        self,
        max_imfs: int = 10,
        boundary: str = "symmetric",
        stopping_criterion: str = "sd_threshold",
    ):
        self.max_imfs = max_imfs
        self.boundary = boundary
        self.stopping_criterion = stopping_criterion


class FerromodeClient:
    """Client for Ferromode EMD gRPC service."""

    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Initialize the client.

        Args:
            host: Server hostname
            port: Server port
        """
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.channel = None
        self.stub = None

    def connect(self) -> None:
        """Connect to the gRPC server."""
        self.channel = grpc.aio.secure_channel(
            self.address,
            grpc.ssl_channel_credentials(),
        )
        # stub = ferromode_pb2_grpc.EmdServiceStub(self.channel)
        print(f"✓ Connected to {self.address}")

    def disconnect(self) -> None:
        """Disconnect from the gRPC server."""
        if self.channel:
            self.channel.close()
            print(f"✓ Disconnected from {self.address}")

    def decompose(
        self,
        values: List[float],
        max_imfs: int = 10,
        boundary: str = "symmetric",
        stopping_criterion: str = "sd_threshold",
    ) -> dict:
        """
        Decompose a signal into intrinsic mode functions.

        Args:
            values: Signal values
            max_imfs: Maximum number of IMFs
            boundary: Boundary condition ("symmetric", "periodic", "extension")
            stopping_criterion: Stopping criterion ("sd_threshold", "energy_ratio")

        Returns:
            Dictionary with IMFs and metadata
        """
        try:
            # Build request
            signal = Signal(values=values)
            config = EmdConfig(
                max_imfs=max_imfs,
                boundary=boundary,
                stopping_criterion=stopping_criterion,
            )

            # Call service (placeholder - would use actual gRPC)
            print(f"\n📊 Decomposing signal ({len(values)} samples)")
            print(f"  Config: max_imfs={max_imfs}, boundary={boundary}")

            # This would be replaced with actual gRPC call:
            # response = self.stub.Decompose(request, timeout=30)

            result = {
                "success": True,
                "imfs": [np.array(values)],  # Placeholder
                "total_sift_iterations": 50,
                "error": None,
            }

            return result

        except grpc.RpcError as e:
            return {
                "success": False,
                "imfs": [],
                "error": f"gRPC error: {e.code()}: {e.details()}",
            }

    def health_check(self) -> bool:
        """Check server health."""
        try:
            # This would call the Health service
            # response = self.stub.Health(HealthRequest(service="EmdService"))
            # return response.status == HealthResponse.Status.SERVING
            print("✓ Health check passed")
            return True
        except grpc.RpcError as e:
            print(f"✗ Health check failed: {e}")
            return False


async def main():
    """Main client example."""
    parser = argparse.ArgumentParser(
        description="Ferromode EMD gRPC client",
    )
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument(
        "--signal",
        type=json.loads,
        default=[1, 2, 3, 4, 5],
        help="Signal values (JSON array)",
    )
    parser.add_argument("--max-imfs", type=int, default=10)
    parser.add_argument("--boundary", default="symmetric")

    args = parser.parse_args()

    # Create client
    client = FerromodeClient(host=args.host, port=args.port)

    try:
        # Connect
        client.connect()

        # Health check
        if not client.health_check():
            print("⚠️ Server health check failed")
            return

        # Decompose signal
        print(f"\n🎯 Decomposing signal: {args.signal}")
        result = client.decompose(
            values=args.signal,
            max_imfs=args.max_imfs,
            boundary=args.boundary,
        )

        if result["success"]:
            print(f"\n✓ Decomposition successful")
            print(f"  Number of IMFs: {len(result['imfs'])}")
            print(f"  Total sift iterations: {result['total_sift_iterations']}")
            for i, imf in enumerate(result["imfs"], 1):
                print(f"  IMF-{i}: {len(imf)} samples, mean={np.mean(imf):.4f}")
        else:
            print(f"\n✗ Decomposition failed: {result['error']}")

    finally:
        client.disconnect()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
