from typing import List, Optional, Tuple, Union, Callable
import einops as E
import torch
from torch import nn
from pydantic import validate_arguments

SizeType = Union[int, Tuple[int, int]]

class UniverSeg(nn.Module):
    def __init__(
        self,
        encoder_blocks: List[Tuple[int, int]],
        decoder_blocks: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        decoder_blocks = decoder_blocks or encoder_blocks[-2::-1]

        # Block 1: Encoder Block 1
        cross_features, conv_features = encoder_blocks[0]
        self.encoder_block1 = CrossBlock((1, 2), cross_features, conv_features)

        # Block 2: Encoder Block 2
        cross_features, conv_features = encoder_blocks[1]
        self.encoder_block2 = CrossBlock(conv_features, cross_features, conv_features)

        # Block 3: Encoder Block 3
        cross_features, conv_features = encoder_blocks[2]
        self.encoder_block3 = CrossBlock(conv_features, cross_features, conv_features)

        # Block 4: Encoder Block 4
        cross_features, conv_features = encoder_blocks[3]
        self.encoder_block4 = CrossBlock(conv_features, cross_features, conv_features)

        # Decoder initialization
        skip_channels = [block[1] for block in encoder_blocks[-2::-1]]

        # Block 5: Decoder Block 1
        cross_features, conv_features = decoder_blocks[0]
        skip_connection = skip_channels[0]
        self.decoder_block1 = CrossBlock(conv_features + skip_connection, cross_features, conv_features)

        # Block 6: Decoder Block 2
        cross_features, conv_features = decoder_blocks[1]
        skip_connection = skip_channels[1]
        self.decoder_block2 = CrossBlock(conv_features + skip_connection, cross_features, conv_features)

        # Block 7: Decoder Block 3
        cross_features, conv_features = decoder_blocks[2]
        skip_connection = skip_channels[2]
        self.decoder_block3 = CrossBlock(conv_features + skip_connection, cross_features, conv_features)

        # Output Convolution
        self.out_conv = ConvOp(conv_features, 1, kernel_size=1)

    def forward(self, target_image, support_images, support_labels):
        # Prepare inputs
        target_tensor = E.rearrange(target_image, "B 1 H W -> B 1 1 H W")
        support_tensor = torch.cat([support_images, support_labels], dim=2)

        # Skip connections
        skip_connections = []

        # Block 1: Encoder Block 1
        target_tensor, support_tensor = self.encoder_block1(target_tensor, support_tensor)
        skip_connections.append((target_tensor, support_tensor))
        target_tensor = vmap(self.downsample, target_tensor)
        support_tensor = vmap(self.downsample, support_tensor)

        # Block 2: Encoder Block 2
        target_tensor, support_tensor = self.encoder_block2(target_tensor, support_tensor)
        skip_connections.append((target_tensor, support_tensor))
        target_tensor = vmap(self.downsample, target_tensor)
        support_tensor = vmap(self.downsample, support_tensor)

        # Block 3: Encoder Block 3
        target_tensor, support_tensor = self.encoder_block3(target_tensor, support_tensor)
        skip_connections.append((target_tensor, support_tensor))
        target_tensor = vmap(self.downsample, target_tensor)
        support_tensor = vmap(self.downsample, support_tensor)

        # Block 4: Encoder Block 4
        target_tensor, support_tensor = self.encoder_block4(target_tensor, support_tensor)

        # Block 5: Decoder Block 1
        target_skip, support_skip = skip_connections.pop()
        target_tensor = torch.cat([vmap(self.upsample, target_tensor), target_skip], dim=2)
        support_tensor = torch.cat([vmap(self.upsample, support_tensor), support_skip], dim=2)
        target_tensor, support_tensor = self.decoder_block1(target_tensor, support_tensor)

        # Block 6: Decoder Block 2
        target_skip, support_skip = skip_connections.pop()
        target_tensor = torch.cat([vmap(self.upsample, target_tensor), target_skip], dim=2)
        support_tensor = torch.cat([vmap(self.upsample, support_tensor), support_skip], dim=2)
        target_tensor, support_tensor = self.decoder_block2(target_tensor, support_tensor)

        # Block 7: Decoder Block 3
        target_skip, support_skip = skip_connections.pop()
        target_tensor = torch.cat([vmap(self.upsample, target_tensor), target_skip], dim=2)
        support_tensor = torch.cat([vmap(self.upsample, support_tensor), support_skip], dim=2)
        target_tensor, support_tensor = self.decoder_block3(target_tensor, support_tensor)

        # Output Convolution
        target_tensor = E.rearrange(target_tensor, "B 1 C H W -> B C H W")
        target_tensor = self.out_conv(target_tensor)

        return target_tensor

def vmap(module: Callable, input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    batch_size, group_size, *_ = input_tensor.shape
    flat_input = E.rearrange(input_tensor, "B G ... -> (B G) ...")
    flat_output = module(flat_input, *args, **kwargs)
    grouped_output = E.rearrange(
        flat_output, "(B G) ... -> B G ...", B=batch_size, G=group_size
    )
    return grouped_output


class CrossConv2d(nn.Conv2d):
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
            combined_channels = sum(input_channels)
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
        batch_size, target_group, *_ = target_tensor.shape
        _, support_group, *_ = support_tensor.shape

        expanded_target = E.repeat(target_tensor, "B T C H W -> B T S C H W", S=support_group)
        expanded_support = E.repeat(support_tensor, "B S C H W -> B T S C H W", T=target_group)

        concatenated_inputs = torch.cat([expanded_target, expanded_support], dim=3)

        flat_inputs = E.rearrange(concatenated_inputs, "B T S C H W -> (B T S) C H W")
        flat_output = super().forward(flat_inputs)

        grouped_output = E.rearrange(
            flat_output, "(B T S) C H W -> B T S C H W", B=batch_size, T=target_group, S=support_group
        )
        return grouped_output


class ConvOp(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: SizeType = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )


class CrossOp(nn.Module):
    @validate_arguments
    def __init__(self, input_channels: SizeType, output_channels: int, kernel_size: SizeType = 3):
        super().__init__()
        self.cross_conv = CrossConv2d(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, target_tensor, support_tensor):
        interactions = self.cross_conv(target_tensor, support_tensor).squeeze(dim=1)
        aggregated_target = interactions.mean(dim=1, keepdims=True)
        return aggregated_target, interactions


class CrossBlock(nn.Module):
    @validate_arguments
    def __init__(
        self,
        input_channels: SizeType,
        cross_features: int,
        conv_features: Optional[int] = None,
    ):
        super().__init__()

        conv_features = conv_features or cross_features

        self.cross = CrossOp(input_channels, cross_features)
        self.target = ConvOp(cross_features, conv_features)
        self.support = ConvOp(cross_features, conv_features)

    def forward(self, target_tensor, support_tensor):
        target_tensor, support_tensor = self.cross(target_tensor, support_tensor)
        target_tensor = vmap(self.target.forward, target_tensor)
        support_tensor = vmap(self.support.forward, support_tensor)
        return target_tensor, support_tensor





def universeg() -> nn.Module:
    model = UniverSeg(
        encoder_blocks=[(64, 64), (64, 64), (64, 64), (64, 64)]
    )
    return model
