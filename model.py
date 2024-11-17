from typing import List, Optional, Tuple, Union, Callable
import einops as E
import torch
from torch import nn
from pydantic import validate_arguments

# Define a type alias for sizes, allowing either an integer or a tuple of two integers
SizeType = Union[int, Tuple[int, int]]

class UniverSeg(nn.Module):
    """
    UniverSeg is a segmentation model that uses an encoder-decoder structure with custom cross-block layers.
    """
    def __init__(
        self,
        encoder_blocks: List[Tuple[int, int]],
        decoder_blocks: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()

        # Downsampling and upsampling operations for resolution management
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # If decoder blocks are not specified, reverse the encoder_blocks for symmetry
        decoder_blocks = decoder_blocks or encoder_blocks[-2::-1]

        # Define encoder blocks (4 levels of encoding)
        cross_features, conv_features = encoder_blocks[0]
        self.encoder_block1 = CrossBlock((1, 2), cross_features, conv_features)

        cross_features, conv_features = encoder_blocks[1]
        self.encoder_block2 = CrossBlock(conv_features, cross_features, conv_features)

        cross_features, conv_features = encoder_blocks[2]
        self.encoder_block3 = CrossBlock(conv_features, cross_features, conv_features)

        cross_features, conv_features = encoder_blocks[3]
        self.encoder_block4 = CrossBlock(conv_features, cross_features, conv_features)

        # Prepare skip connection channels for the decoder
        skip_channels = [block[1] for block in encoder_blocks[-2::-1]]

        # Define decoder blocks (symmetrical to encoder blocks)
        cross_features, conv_features = decoder_blocks[0]
        skip_connection = skip_channels[0]
        self.decoder_block1 = CrossBlock(conv_features + skip_connection, cross_features, conv_features)

        cross_features, conv_features = decoder_blocks[1]
        skip_connection = skip_channels[1]
        self.decoder_block2 = CrossBlock(conv_features + skip_connection, cross_features, conv_features)

        cross_features, conv_features = decoder_blocks[2]
        skip_connection = skip_channels[2]
        self.decoder_block3 = CrossBlock(conv_features + skip_connection, cross_features, conv_features)

        # Define the output convolution to reduce channels to the desired output
        self.out_conv = ConvOp(conv_features, 1, kernel_size=1)

    def forward(self, target_image, support_images, support_labels):
        """
        Forward pass for the UniverSeg model, involving encoding and decoding.

        Args:
            target_image: The image to segment.
            support_images: Images providing context for segmentation.
            support_labels: Labels corresponding to support_images.

        Returns:
            target_tensor: The segmented output.
        """
        # Reshape inputs and combine support images with their labels
        target_tensor = E.rearrange(target_image, "B 1 H W -> B 1 1 H W")
        support_tensor = torch.cat([support_images, support_labels], dim=2)

        # List to store skip connections from encoder blocks
        skip_connections = []

        # Encoder Blocks (capture hierarchical features)
        target_tensor, support_tensor = self.encoder_block1(target_tensor, support_tensor)
        skip_connections.append((target_tensor, support_tensor))
        target_tensor = vmap(self.downsample, target_tensor)
        support_tensor = vmap(self.downsample, support_tensor)

        target_tensor, support_tensor = self.encoder_block2(target_tensor, support_tensor)
        skip_connections.append((target_tensor, support_tensor))
        target_tensor = vmap(self.downsample, target_tensor)
        support_tensor = vmap(self.downsample, support_tensor)

        target_tensor, support_tensor = self.encoder_block3(target_tensor, support_tensor)
        skip_connections.append((target_tensor, support_tensor))
        target_tensor = vmap(self.downsample, target_tensor)
        support_tensor = vmap(self.downsample, support_tensor)

        target_tensor, support_tensor = self.encoder_block4(target_tensor, support_tensor)

        # Decoder Blocks (reconstruct using hierarchical features)
        target_skip, support_skip = skip_connections.pop()
        target_tensor = torch.cat([vmap(self.upsample, target_tensor), target_skip], dim=2)
        support_tensor = torch.cat([vmap(self.upsample, support_tensor), support_skip], dim=2)
        target_tensor, support_tensor = self.decoder_block1(target_tensor, support_tensor)

        target_skip, support_skip = skip_connections.pop()
        target_tensor = torch.cat([vmap(self.upsample, target_tensor), target_skip], dim=2)
        support_tensor = torch.cat([vmap(self.upsample, support_tensor), support_skip], dim=2)
        target_tensor, support_tensor = self.decoder_block2(target_tensor, support_tensor)

        target_skip, support_skip = skip_connections.pop()
        target_tensor = torch.cat([vmap(self.upsample, target_tensor), target_skip], dim=2)
        support_tensor = torch.cat([vmap(self.upsample, support_tensor), support_skip], dim=2)
        target_tensor, support_tensor = self.decoder_block3(target_tensor, support_tensor)

        # Output convolution to generate the final segmentation map
        target_tensor = E.rearrange(target_tensor, "B 1 C H W -> B C H W")
        target_tensor = self.out_conv(target_tensor)

        return target_tensor


class CrossConv2d(nn.Conv2d):
    """
    Custom 2D convolution layer for cross-attention between target and support tensors.
    """
    @validate_arguments
    def __init__(
        self,
        input_channels: SizeType,
        output_channels: int,
        kernel_size: SizeType,
        stride: SizeType = 1,
        padding: SizeType = 0,
        dilation: SizeType = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:

        if isinstance(input_channels, (list, tuple)):
            combined_channels = sum(input_channels)  # Combine input channels from multiple sources
        else:
            combined_channels = 2 * input_channels

        super().__init__(
            in_channels=combined_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, target_tensor: torch.Tensor, support_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution after merging target and support tensors.

        Args:
            target_tensor: Tensor representing the target data.
            support_tensor: Tensor representing the support data.

        Returns:
            grouped_output: Processed output tensor.
        """
        batch_size, target_group, *_ = target_tensor.shape
        _, support_group, *_ = support_tensor.shape

        # Expand target and support tensors to align dimensions
        expanded_target = E.repeat(target_tensor, "B T C H W -> B T S C H W", S=support_group)
        expanded_support = E.repeat(support_tensor, "B S C H W -> B T S C H W", T=target_group)

        # Concatenate expanded tensors and flatten for convolution
        concatenated_inputs = torch.cat([expanded_target, expanded_support], dim=3)
        flat_inputs = E.rearrange(concatenated_inputs, "B T S C H W -> (B T S) C H W")
        flat_output = super().forward(flat_inputs)

        # Rearrange output back to grouped format
        grouped_output = E.rearrange(
            flat_output, "(B T S) C H W -> B T S C H W", B=batch_size, T=target_group, S=support_group
        )
        return grouped_output


class ConvOp(nn.Sequential):
    """
    A simple wrapper around nn.Conv2d to encapsulate a standard convolution operation.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: SizeType = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Ensures the output size matches the input size
            padding_mode="zeros",
            bias=True,
        )


class CrossOp(nn.Module):
    """
    CrossOp module for interacting between target and support tensors using CrossConv2d.
    """
    @validate_arguments
    def __init__(self, input_channels: SizeType, output_channels: int, kernel_size: SizeType = 3):
        super().__init__()
        self.cross_conv = CrossConv2d(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Keeps spatial dimensions consistent
        )

    def forward(self, target_tensor, support_tensor):
        """
        Forward pass for cross-operation to extract and aggregate features.

        Args:
            target_tensor: Target tensor to process.
            support_tensor: Support tensor for feature interactions.

        Returns:
            aggregated_target: Aggregated features for the target tensor.
            interactions: Interaction features between target and support tensors.
        """
        # Compute interactions between target and support tensors
        interactions = self.cross_conv(target_tensor, support_tensor).squeeze(dim=1)

        # Aggregate features across support tensors
        aggregated_target = interactions.mean(dim=1, keepdims=True)

        return aggregated_target, interactions


class CrossBlock(nn.Module):
    """
    CrossBlock combines cross-tensor interactions and subsequent processing with convolution layers.
    """
    @validate_arguments
    def __init__(
        self,
        input_channels: SizeType,
        cross_features: int,
        conv_features: Optional[int] = None,
    ):
        super().__init__()

        # Use cross_features as default for conv_features if not explicitly provided
        conv_features = conv_features or cross_features

        # Define cross-operation and convolution layers
        self.cross = CrossOp(input_channels, cross_features)
        self.target = ConvOp(cross_features, conv_features)
        self.support = ConvOp(cross_features, conv_features)

    def forward(self, target_tensor, support_tensor):
        """
        Forward pass for a single cross block.

        Args:
            target_tensor: Input tensor for the target.
            support_tensor: Input tensor for the support data.

        Returns:
            target_tensor: Processed target tensor.
            support_tensor: Processed support tensor.
        """
        # Cross-operation between target and support tensors
        target_tensor, support_tensor = self.cross(target_tensor, support_tensor)

        # Process the resulting tensors with separate convolution layers
        target_tensor = vmap(self.target.forward, target_tensor)
        support_tensor = vmap(self.support.forward, support_tensor)

        return target_tensor, support_tensor


def vmap(module: Callable, input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Vectorized mapping utility for applying a module across batched inputs.

    Args:
        module: The module to apply (e.g., a layer or operation).
        input_tensor: The input tensor with batch and group dimensions.
        *args, **kwargs: Additional arguments for the module.

    Returns:
        grouped_output: The output tensor after applying the module across batches.
    """
    batch_size, group_size, *_ = input_tensor.shape

    # Flatten batch and group dimensions for efficient processing
    flat_input = E.rearrange(input_tensor, "B G ... -> (B G) ...")
    flat_output = module(flat_input, *args, **kwargs)

    # Rearrange back to the original grouped format
    grouped_output = E.rearrange(
        flat_output, "(B G) ... -> B G ...", B=batch_size, G=group_size
    )
    return grouped_output


def universeg() -> nn.Module:
    """
    Factory function for creating a UniverSeg model with default configurations.

    Returns:
        model: An instance of the UniverSeg model.
    """
    model = UniverSeg(
        encoder_blocks=[(64, 64), (64, 64), (64, 64), (64, 64)]  # Default encoder configuration
    )
    return model
