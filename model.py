from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import einops as E
import torch
from torch import nn
import torch.nn.init as init

size2t = Union[int, Tuple[int, int]]


def vmap(module: Callable, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    batch_size, group_size, *_ = x.shape
    grouped_input = E.rearrange(x, "B S ... -> (B S) ...")
    grouped_output = module(grouped_input, *args, **kwargs)
    output = E.rearrange(
        grouped_output, "(B S) ... -> B S ...", B=batch_size, S=group_size
    )
    return output


class Vmap(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.vmapped = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vmap(self.vmapped, x)


class CrossConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: size2t,
        out_channels: int,
        kernel_size: size2t,
        stride: size2t = 1,
        padding: size2t = 0,
        dilation: size2t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        nonlinearity: Optional[str] = "leaky_relu",
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[None, float, int] = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        if isinstance(in_channels, (list, tuple)):
            concat_channels = sum(in_channels)
        else:
            concat_channels = 2 * in_channels

        super().__init__(
            in_channels=concat_channels,
            out_channels=out_channels,
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

        # Initialize weights and biases
        self.initialize_weights(init_distribution, nonlinearity, init_bias)

    def initialize_weights(
        self, init_distribution: Optional[str], nonlinearity: Optional[str], init_bias: Union[None, float, int]
    ):
        if nonlinearity is not None:
            nonlinearity = nonlinearity.lower().replace("leakyrelu", "leaky_relu")

        if init_distribution == "kaiming_normal":
            init.kaiming_normal_(self.weight, nonlinearity=nonlinearity)
        elif init_distribution == "kaiming_uniform":
            init.kaiming_uniform_(self.weight, nonlinearity=nonlinearity)
        elif init_distribution == "xavier_normal":
            init.xavier_normal_(self.weight)
        elif init_distribution == "xavier_uniform":
            init.xavier_uniform_(self.weight)
        else:
            init.kaiming_normal_(self.weight, nonlinearity="relu")  # Default to ReLU

        if self.bias is not None and init_bias is not None:
            init.constant_(self.bias, init_bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise convolution between all elements of x and y.

        Args:
            x: Tensor of size (B, Sx, Cx, H, W).
            y: Tensor of size (B, Sy, Cy, H, W).

        Returns:
            Tensor resulting from the cross-convolution.
        """
        B, Sx, _, H, W = x.shape
        _, Sy, _, _, _ = y.shape

        xs = E.repeat(x, "B Sx Cx H W -> B Sx Sy Cx H W", Sy=Sy)
        ys = E.repeat(y, "B Sy Cy H W -> B Sx Sy Cy H W", Sx=Sx)

        xy = torch.cat([xs, ys], dim=3)  # Concatenate along channel dimension

        batched_xy = E.rearrange(xy, "B Sx Sy C H W -> (B Sx Sy) C H W")
        batched_output = super().forward(batched_xy)

        output = E.rearrange(
            batched_output, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy
        )
        return output



class ConvOp(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: size2t = 3,
        nonlinearity: Optional[str] = "leaky_relu",
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[None, float, int] = 0.0,
    ):
        super().__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        # Normalize nonlinearity names
        if nonlinearity is not None:
            nonlinearity = nonlinearity.lower().replace("leakyrelu", "leaky_relu")

        # Initialize weights and biases
        if init_distribution == "kaiming_normal":
            init.kaiming_normal_(conv.weight, nonlinearity=nonlinearity)
        elif init_distribution == "kaiming_uniform":
            init.kaiming_uniform_(conv.weight, nonlinearity=nonlinearity)
        elif init_distribution == "xavier_normal":
            init.xavier_normal_(conv.weight)
        elif init_distribution == "xavier_uniform":
            init.xavier_uniform_(conv.weight)
        else:
            init.kaiming_normal_(conv.weight, nonlinearity="relu")  # Default to ReLU

        if conv.bias is not None and init_bias is not None:
            init.constant_(conv.bias, init_bias)

        self.add_module("conv", conv)


class CrossOp(nn.Module):
    def __init__(
        self,
        in_channels: size2t,
        out_channels: int,
        kernel_size: size2t = 3,
        nonlinearity: Optional[str] = "leaky_relu",
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[None, float, int] = 0.0,
    ):
        super().__init__()
        self.cross_conv = CrossConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            nonlinearity=nonlinearity,
            init_distribution=init_distribution,
            init_bias=init_bias,
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)
        new_target = interaction.mean(dim=1, keepdims=True)
        return new_target, interaction


class CrossBlock(nn.Module):
    def __init__(
        self,
        in_channels: size2t,
        cross_features: int,
        conv_features: Optional[int] = None,
        nonlinearity: Optional[str] = "leaky_relu",
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[None, float, int] = 0.0,
    ):
        super().__init__()

        conv_features = conv_features or cross_features

        self.cross = CrossOp(
            in_channels,
            cross_features,
            nonlinearity=nonlinearity,
            init_distribution=init_distribution,
            init_bias=init_bias,
        )
        self.target = Vmap(
            ConvOp(
                cross_features,
                conv_features,
                nonlinearity=nonlinearity,
                init_distribution=init_distribution,
                init_bias=init_bias,
            )
        )
        self.support = Vmap(
            ConvOp(
                cross_features,
                conv_features,
                nonlinearity=nonlinearity,
                init_distribution=init_distribution,
                init_bias=init_bias,
            )
        )

    def forward(self, target, support):
        target, support = self.cross(target, support)
        target = self.target(target)
        support = self.support(support)
        return target, support


class UniverSeg(nn.Module):
    def __init__(
        self,
        encoder_blocks: List[Tuple[int, int]],
        decoder_blocks: Optional[List[Tuple[int, int]]] = None,
        nonlinearity: Optional[str] = "leaky_relu",
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[None, float, int] = 0.0,
    ):
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        decoder_blocks = decoder_blocks or encoder_blocks[-2::-1]

        in_ch = (1, 2)
        out_channels = 1

        # Encoder
        skip_outputs = []
        for (cross_ch, conv_ch) in encoder_blocks:
            block = CrossBlock(
                in_ch,
                cross_ch,
                conv_ch,
                nonlinearity=nonlinearity,
                init_distribution=init_distribution,
                init_bias=init_bias,
            )
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(
                in_ch + skip_ch,
                cross_ch,
                conv_ch,
                nonlinearity=nonlinearity,
                init_distribution=init_distribution,
                init_bias=init_bias,
            )
            in_ch = conv_ch
            self.dec_blocks.append(block)

        self.out_conv = ConvOp(
            in_ch, out_channels, kernel_size=1, nonlinearity=nonlinearity
        )

    def forward(self, target_image, support_images, support_labels):
        target = E.rearrange(target_image, "B 1 H W -> B 1 1 H W")
        support = torch.cat([support_images, support_labels], dim=2)

        pass_through = []

        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.enc_blocks) - 1:
                break
            pass_through.append((target, support))
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        for decoder_block in self.dec_blocks:
            target_skip, support_skip = pass_through.pop()
            target = torch.cat([vmap(self.upsample, target), target_skip], dim=2)
            support = torch.cat([vmap(self.upsample, support), support_skip], dim=2)
            target, support = decoder_block(target, support)

        target = E.rearrange(target, "B 1 C H W -> B C H W")
        target = self.out_conv(target)

        return target


def universeg(
    nonlinearity: Optional[str] = "leaky_relu",
    init_distribution: Optional[str] = "kaiming_normal",
    init_bias: Union[None, float, int] = 0.0,
) -> nn.Module:
    return UniverSeg(
        encoder_blocks=[(64, 64), (64, 64), (64, 64), (64, 64)],
        nonlinearity=nonlinearity,
        init_distribution=init_distribution,
        init_bias=init_bias,
    )
