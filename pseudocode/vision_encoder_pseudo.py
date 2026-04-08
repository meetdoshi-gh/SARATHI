"""
vision_encoder_pseudo.py
------------------------
CNN vision encoder: lightweight 3-block convolutional network designed
for grayscale, memory-constrained RL training.

Abstraction level: class/method skeletons with docstrings.
This file is illustrative pseudocode — not executable.
"""

import torch.nn as nn


class VisionEncoderBase(nn.Module):
    """
    Abstract base class for all vision encoders used in the SARATHI pipeline.

    Subclasses implement _forward_impl() for architecture-specific logic.
    The base class handles device placement and optional image resizing.

    Parameters
    ----------
    num_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    use_resizer : bool
        If True, prepend a learned image resizer before the encoder.
    target_size : tuple
        (H, W) target resolution if use_resizer is True.
    """

    def __init__(self, num_channels: int, use_resizer: bool, target_size: tuple) -> None:
        ...

    def forward(self, x: "Tensor") -> "Tensor":
        """
        Run the full encoder forward pass.

        Ensures input is a float32 tensor on the correct device,
        optionally resizes, then delegates to _forward_impl().

        Parameters
        ----------
        x : Tensor
            Input image batch, shape (B, C, H, W).

        Returns
        -------
        Tensor
            Flat feature vector, shape (B, feature_dim).
        """
        ...

    def _forward_impl(self, x: "Tensor") -> "Tensor":
        """
        Architecture-specific forward pass. Implemented by subclasses.
        """
        raise NotImplementedError


class CNNEncoderGroupNorm(VisionEncoderBase):
    """
    Lightweight 3-block CNN encoder with GroupNorm.

    Designed for single-channel (grayscale) input at 120×160 resolution.
    GroupNorm is used instead of BatchNorm to remain batch-size independent,
    which is critical for RL training where batch sizes are small.

    Architecture summary
    --------------------
    Block 1: Conv2d(C→32, k=8, s=4) → GroupNorm(8 groups) → ReLU → MaxPool(2)
             Output: 14×19×32
    Block 2: Conv2d(32→64, k=4, s=2) → GroupNorm(8 groups) → ReLU → MaxPool(2)
             Output: 3×4×64
    Block 3: Conv2d(64→64, k=3, s=1) → GroupNorm(8 groups) → ReLU
             Output: 1×2×64 → Flatten → 128-dim

    Total parameters: ~72,000 per encoder copy.

    Parameters
    ----------
    num_channels : int
        Input channels. 1 for grayscale.
    """

    def __init__(self, num_channels: int = 1) -> None:
        """Build conv blocks with GroupNorm normalization."""
        ...

    def _forward_impl(self, x: "Tensor") -> "Tensor":
        """
        Pass input through the three conv blocks and flatten output.

        Returns
        -------
        Tensor
            Shape (B, 128).
        """
        ...


class InputEncoderFusion(nn.Module):
    """
    Fuses stereo left/right visual features with the navigation info vector.

    Runs the shared vision encoder on each camera image independently,
    then concatenates: [left_feat | right_feat | nav_info].

    Result: a (B, 2*vision_dim + nav_dim) fused observation vector.
    In the standard configuration: (B, 128 + 128 + 10) = (B, 266).

    Parameters
    ----------
    img_res : tuple
        (H, W) input image resolution.
    nav_dim : int
        Dimensionality of the navigation info vector.
    num_channels : int
        Number of input image channels.
    vision_encoder_cls : type
        Vision encoder class to instantiate (shared weights for left/right).
    """

    def __init__(
        self,
        img_res: tuple,
        nav_dim: int,
        num_channels: int,
        vision_encoder_cls,
    ) -> None:
        """
        Instantiate a single shared vision encoder.
        Infer output dimensionality using a dummy forward pass.
        """
        ...

    def forward(
        self,
        left_img: "Tensor",
        right_img: "Tensor",
        nav_info: "Tensor",
    ) -> "Tensor":
        """
        Encode both images with the shared encoder, then concatenate with nav.

        Parameters
        ----------
        left_img, right_img : Tensor
            Shape (B, C, H, W).
        nav_info : Tensor
            Shape (B, nav_dim).

        Returns
        -------
        Tensor
            Fused features, shape (B, 2*vision_dim + nav_dim).
        """
        ...
