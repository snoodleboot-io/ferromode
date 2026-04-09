#!/usr/bin/env python3
"""
Kubernetes operator for Ferromode EMD decomposition jobs using kopf.

This operator watches for EMDJob resources and manages the lifecycle of
decomposition tasks across a Kubernetes cluster.
"""

import kopf
import kubernetes
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

# Operator configuration
EMD_IMAGE = os.environ.get("EMD_IMAGE", "ferromode:latest")
FERROMODE_NAMESPACE = "ferromode-system"


@kopf.on.event(
    "ferromode.io",
    "v1alpha1",
    "emdjobs",
    annotations={
        kopf.RESUME_ANNOTATION: kopf.RESUME,
    },
)
def monitor_emdjob(event, **kwargs):
    """Monitor EMDJob events and log them."""
    logger.info(
        f"EMDJob event: {event['type']} - {event['object']['metadata']['name']}"
    )


@kopf.on.event(
    "ferromode.io",
    "v1alpha1",
    "emdjobs",
    initial=False,
)
async def emdjob_created(
    name: str,
    namespace: str,
    spec: Dict[str, Any],
    status: Dict[str, Any],
    **kwargs,
):
    """Handle EMDJob creation."""
    logger.info(f"EMDJob created: {name} in {namespace}")

    try:
        # Validate spec
        if "signal" not in spec:
            raise kopf.PermanentError("signal configuration is required")

        # Create a Job for each parallel trial
        parallelism = spec.get("parallelism", 4)
        timeout_str = spec.get("timeout", "1h")

        # Parse timeout
        timeout_seconds = parse_timeout(timeout_str)

        # Create Job resources
        await create_job_resources(
            name=name,
            namespace=namespace,
            spec=spec,
            parallelism=parallelism,
            timeout_seconds=timeout_seconds,
        )

        # Update status to Running
        kopf.patch(
            {
                "apiVersion": "ferromode.io/v1alpha1",
                "kind": "EMDJob",
                "metadata": {"name": name, "namespace": namespace},
            },
            {
                "status": {
                    "phase": "Running",
                    "startedAt": datetime.utcnow().isoformat() + "Z",
                }
            },
        )

    except Exception as e:
        logger.error(f"Error creating EMDJob: {e}")
        kopf.patch(
            {
                "apiVersion": "ferromode.io/v1alpha1",
                "kind": "EMDJob",
                "metadata": {"name": name, "namespace": namespace},
            },
            {
                "status": {
                    "phase": "Failed",
                    "message": str(e),
                }
            },
        )
        raise


@kopf.on.event(
    "batch",
    "v1",
    "jobs",
    labels={"app": "emd-decomposition"},
    initial=False,
)
async def job_completed(name: str, namespace: str, status: Dict[str, Any], **kwargs):
    """Handle Job completion and aggregate results."""
    logger.info(f"Job completed: {name}")

    if status.get("succeeded") == 1:
        logger.info(f"Job {name} succeeded")
        # Aggregate results from completed job
        await aggregate_job_results(name, namespace)
    elif status.get("failed"):
        logger.error(f"Job {name} failed")


@kopf.on.delete(
    "ferromode.io",
    "v1alpha1",
    "emdjobs",
)
async def emdjob_deleted(name: str, namespace: str, **kwargs):
    """Handle EMDJob deletion and cleanup."""
    logger.info(f"EMDJob deleted: {name}")

    # Delete associated Job resources
    api = kubernetes.client.BatchV1Api()

    try:
        jobs = api.list_namespaced_job(
            namespace,
            label_selector=f"emd-job={name}",
        )

        for job in jobs.items:
            api.delete_namespaced_job(
                job.metadata.name,
                namespace,
                propagation_policy="Background",
            )
            logger.info(f"Deleted job: {job.metadata.name}")

    except Exception as e:
        logger.error(f"Error deleting job resources: {e}")


async def create_job_resources(
    name: str,
    namespace: str,
    spec: Dict[str, Any],
    parallelism: int,
    timeout_seconds: int,
) -> None:
    """Create Kubernetes Job resources for EMD decomposition."""
    api = kubernetes.client.BatchV1Api()

    signal_config = spec.get("signal", {})
    emd_config = spec.get("config", {})

    # Create Job spec
    job_spec = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"{name}-{parallelism}",
            "namespace": namespace,
            "labels": {
                "app": "emd-decomposition",
                "emd-job": name,
            },
        },
        "spec": {
            "parallelism": parallelism,
            "completions": parallelism,
            "activeDeadlineSeconds": timeout_seconds,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "emd-decomposition",
                        "emd-job": name,
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "name": "emd-worker",
                            "image": EMD_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "env": [
                                {
                                    "name": "SIGNAL_URL",
                                    "value": signal_config.get("url", ""),
                                },
                                {
                                    "name": "SIGNAL_SOURCE",
                                    "value": signal_config.get("source", "parquet"),
                                },
                                {
                                    "name": "MAX_IMFS",
                                    "value": str(emd_config.get("maxImfs", 10)),
                                },
                                {
                                    "name": "BOUNDARY",
                                    "value": emd_config.get("boundary", "symmetric"),
                                },
                                {
                                    "name": "RESULT_URL",
                                    "value": spec.get(
                                        "resultUrl",
                                        f"s3://emd-results/{namespace}/{name}",
                                    ),
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "1",
                                    "memory": "2Gi",
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": "4Gi",
                                },
                            },
                        }
                    ],
                    "restartPolicy": "OnFailure",
                    "backoffLimit": 3,
                },
            },
        },
    }

    try:
        api.create_namespaced_job(namespace, job_spec)
        logger.info(f"Created job: {name}")
    except kubernetes.client.exceptions.ApiException as e:
        logger.error(f"Error creating job: {e}")
        raise


async def aggregate_job_results(job_name: str, namespace: str) -> None:
    """Aggregate results from completed decomposition job."""
    logger.info(f"Aggregating results from job: {job_name}")

    # Implementation would fetch results from storage (S3, GCS, etc)
    # and update the parent EMDJob status

    # For now, this is a placeholder
    pass


def parse_timeout(timeout_str: str) -> int:
    """Parse timeout string (e.g., '30m', '1h') to seconds."""
    import re

    match = re.match(r"(\d+)([smh])", timeout_str.lower())
    if not match:
        return 3600  # default 1 hour

    value, unit = match.groups()
    value = int(value)

    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600

    return 3600


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting Ferromode EMDJob operator")
    kopf.run()
