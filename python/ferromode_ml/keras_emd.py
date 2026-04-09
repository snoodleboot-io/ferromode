"""Keras layer for differentiable EMD.

This module provides a Keras layer that integrates differentiable EMD
into TensorFlow/Keras models, enabling end-to-end gradient-based training.

Classes:
    DifferentiableEMDLayer: Keras layer for EMD decomposition
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore

from .rust_bridge import call_rust_emd_backward, call_rust_emd_forward


class DifferentiableEMDLayer(tf.keras.layers.Layer):  # type: ignore
    """Keras layer for differentiable EMD with automatic differentiation.

    This layer decomposes input signals into Intrinsic Mode Functions (IMFs)
    and a residue component. Gradients flow backward through the decomposition
    using implicit differentiation, enabling EMD to be used in deep learning
    pipelines.

    The layer accepts batches of signals and returns a batch of decomposed IMFs.

    Args:
        max_imfs: Maximum number of IMFs to extract (int or None)
            If None, determined automatically by stopping criteria.
            Default: None
        boundary: Boundary extension method ("mirror", "periodic", "symm")
            Default: "mirror"
        sifting_iterations: Maximum sifting iterations per IMF (int)
            Default: 100
        sifting_tol: Convergence tolerance for sifting loop (float)
            Default: 1e-6
        name: Layer name (str)
            Default: "differentiable_emd"
        **kwargs: Additional Keras layer arguments

    Input Shape:
        - If input_shape is provided as arg:
            (batch_size, signal_length)
        - Dynamically:
            (batch_size, signal_length) or (signal_length,)

    Output Shape:
        (batch_size, num_imfs, signal_length) if batch dimension present
        (num_imfs, signal_length) if no batch dimension

    Examples:
        >>> # Simple usage in Keras Sequential model
        >>> layer = DifferentiableEMDLayer(max_imfs=5)
        >>> inputs = tf.random.normal((32, 100))  # 32 signals of length 100
        >>> outputs = layer(inputs)  # (32, num_imfs, 100)

        >>> # In a model for signal classification
        >>> inputs = tf.keras.Input(shape=(256,))  # 256-length signals
        >>> x = DifferentiableEMDLayer(max_imfs=5, name="emd")(inputs)
        >>> x = tf.keras.layers.Reshape((x.shape[1] * x.shape[2],))(x)
        >>> x = tf.keras.layers.Dense(64, activation="relu")(x)
        >>> x = tf.keras.layers.Dense(10, activation="softmax")(x)
        >>> model = tf.keras.Model(inputs, x)

    Notes:
        - The backward pass uses implicit differentiation, not backprop
          through the sifting loop, for numerical stability.
        - Batch processing: signals are processed individually, then stacked.
        - The number of extracted IMFs may vary per signal; padding/truncation
          is handled automatically.
    """

    def __init__(
        self,
        max_imfs: Optional[int] = None,
        boundary: str = "mirror",
        sifting_iterations: int = 100,
        sifting_tol: float = 1e-6,
        name: str = "differentiable_emd",
        **kwargs: Any,
    ):
        """Initialize DifferentiableEMDLayer."""
        if tf is None:
            raise ImportError("TensorFlow is required for DifferentiableEMDLayer")

        super().__init__(name=name, **kwargs)

        self.max_imfs = max_imfs
        self.boundary = boundary
        self.sifting_iterations = sifting_iterations
        self.sifting_tol = sifting_tol

        # Build EMD configuration
        self.emd_config: Dict[str, Any] = {
            "boundary": boundary,
            "sifting_iterations": sifting_iterations,
            "sifting_tol": sifting_tol,
        }
        if max_imfs is not None:
            self.emd_config["max_imfs"] = max_imfs

        # Cache for forward pass contexts (for backward pass)
        self._contexts: Dict[int, Dict[str, Any]] = {}
        self._context_counter = 0

    def call(
        self,
        inputs: tf.Tensor,  # type: ignore
        training: Optional[bool] = None,
    ) -> tf.Tensor:  # type: ignore
        """Forward pass: decompose batch of signals.

        Args:
            inputs: Batch of signals
                Shape: (batch_size, signal_length) or (signal_length,)
            training: Whether in training mode (for batch norm, dropout, etc.)

        Returns:
            IMFs for all signals in batch
            Shape: (batch_size, num_imfs, signal_length) if batch present
            Shape: (num_imfs, signal_length) if no batch
        """
        # Ensure batch dimension
        input_shape = tf.shape(inputs)
        is_batched = inputs.shape.rank == 2

        if not is_batched:
            # Add batch dimension
            inputs = tf.expand_dims(inputs, axis=0)

        batch_size = tf.shape(inputs)[0]
        signal_length = tf.shape(inputs)[1]

        # Use custom_gradient for backward pass
        @tf.custom_gradient
        def emd_forward_and_backward(batch_signals: tf.Tensor) -> Tuple[tf.Tensor, Any]:  # type: ignore
            """Forward and backward pass for EMD."""
            # Process each signal in batch
            all_imfs: List[np.ndarray] = []
            all_contexts: List[Dict[str, Any]] = []
            max_imfs_found = 0

            for i in range(batch_size.numpy()):
                signal = batch_signals[i].numpy()

                # Call Rust forward pass
                imfs, context = call_rust_emd_forward(signal, self.emd_config, "numpy")

                all_imfs.append(imfs)
                all_contexts.append(context)

                max_imfs_found = max(max_imfs_found, imfs.shape[0])

            # Pad all IMFs to same number
            padded_imfs: List[np.ndarray] = []
            for imfs in all_imfs:
                if imfs.shape[0] < max_imfs_found:
                    # Pad with zeros
                    padding = np.zeros(
                        (max_imfs_found - imfs.shape[0], imfs.shape[1]),
                        dtype=np.float64,
                    )
                    imfs = np.vstack([imfs, padding])
                padded_imfs.append(imfs)

            # Stack into batch tensor
            output_np = np.stack(padded_imfs, axis=0)
            output_tf = tf.constant(output_np, dtype=tf.float32)

            # Store contexts for backward pass
            context_id = self._context_counter
            self._context_counter += 1
            self._contexts[context_id] = {
                "contexts": all_contexts,
                "original_imf_counts": [imfs.shape[0] for imfs in all_imfs],
            }

            def grad(upstream_grad: tf.Tensor) -> tf.Tensor:  # type: ignore
                """Backward pass using implicit differentiation."""
                upstream_np = upstream_grad.numpy().astype(np.float64)

                # Compute gradients for each signal
                grad_signals: List[np.ndarray] = []

                for i in range(batch_size.numpy()):
                    context = all_contexts[i]
                    signal = batch_signals[i].numpy().astype(np.float64)

                    # Extract upstream gradient for this signal
                    grad_imfs_i = upstream_np[i, :, :]

                    # Call Rust backward pass
                    grad_signal = call_rust_emd_backward(
                        grad_imfs_i,
                        signal,
                        context,
                        "numpy",
                    )

                    grad_signals.append(grad_signal)

                # Stack gradients
                grad_batch = np.stack(grad_signals, axis=0)
                return tf.constant(grad_batch, dtype=tf.float32)

            return output_tf, grad

        # Call custom gradient function
        outputs = emd_forward_and_backward(inputs)

        # Remove batch dimension if input wasn't batched
        if not is_batched:
            outputs = tf.squeeze(outputs, axis=0)

        return outputs

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        if len(input_shape) == 1:
            # Single signal, no batch
            signal_length = input_shape[0]
            # Output: (num_imfs, signal_length)
            # Note: num_imfs is not known until runtime
            return (None, signal_length)

        elif len(input_shape) == 2:
            # Batched signals
            batch_size = input_shape[0]
            signal_length = input_shape[1]
            # Output: (batch_size, num_imfs, signal_length)
            return (batch_size, None, signal_length)

        else:
            raise ValueError(f"Invalid input shape: {input_shape}")

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary of layer config
        """
        config = super().get_config()
        config.update(
            {
                "max_imfs": self.max_imfs,
                "boundary": self.boundary,
                "sifting_iterations": self.sifting_iterations,
                "sifting_tol": self.sifting_tol,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DifferentiableEMDLayer":
        """Create layer from configuration.

        Args:
            config: Configuration dictionary from get_config()

        Returns:
            New DifferentiableEMDLayer instance
        """
        return cls(**config)
