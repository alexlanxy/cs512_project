from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import einops as E
import torch
from torch import nn
from torch.nn import init

import warnings

from pydantic import validate_arguments

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
    @validate_arguments
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, Sx, *_ = x.shape
        _, Sy, *_ = y.shape

        xs = E.repeat(x, "B Sx Cx H W -> B Sx Sy Cx H W", Sy=Sy)
        ys = E.repeat(y, "B Sy Cy H W -> B Sx Sy Cy H W", Sx=Sx)

        xy = torch.cat([xs, ys], dim=3,)

        batched_xy = E.rearrange(xy, "B Sx Sy C2 H W -> (B Sx Sy) C2 H W")
        batched_output = super().forward(batched_xy)

        output = E.rearrange(
            batched_output, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy
        )
        return output


class ConvOp(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: size2t = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

class CrossOp(nn.Module):
    @validate_arguments
    def __init__(self, in_channels: size2t, out_channels: int, kernel_size: size2t = 3):
        super().__init__()
        self.cross_conv = CrossConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)
        new_target = interaction.mean(dim=1, keepdims=True)
        return new_target, interaction

class CrossBlock(nn.Module):
    @validate_arguments
    def __init__(
        self,
        in_channels: size2t,
        cross_features: int,
        conv_features: Optional[int] = None,
        cross_kws: Optional[Dict[str, Any]] = None,
        conv_kws: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        conv_features = conv_features or cross_features
        cross_kws = cross_kws or {}
        conv_kws = conv_kws or {}

        self.cross = CrossOp(in_channels, cross_features, **cross_kws)
        self.target = Vmap(ConvOp(cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(cross_features, conv_features, **conv_kws))

    def forward(self, target, support):
        target, support = self.cross(target, support)
        target = self.target(target)
        support = self.support(support)
        return target, support


class UniverSeg(nn.Module):
    def __init__(
        self,
        encoder_blocks: List[Tuple[int, int]],  # Expecting a list of 2-tuples
        decoder_blocks: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        decoder_blocks = decoder_blocks or encoder_blocks[-2::-1]

        # Remove nonlinearity from block_kws
        block_kws = dict(cross_kws={})

        in_ch = (1, 2)
        out_channels = 1
        out_activation = None

        # Encoder
        skip_outputs = []
        for (cross_ch, conv_ch) in encoder_blocks:
            block = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(in_ch + skip_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.dec_blocks.append(block)

        self.out_conv = ConvOp(
            in_ch, out_channels, kernel_size=1
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


def universeg() -> nn.Module:
    model = UniverSeg(
        encoder_blocks=[(64, 64), (64, 64), (64, 64), (64, 64)]
    )
    return model