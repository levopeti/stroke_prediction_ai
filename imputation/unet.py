"""
Full assembly of the parts to form the complete network
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
from typing import Optional

import torch
from torch.nn import Module
from imputation.unet_utils import DoubleConv, Down, Up, OutConv


class UNet(Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64, mid_channels=32)
        self.inc_mask = DoubleConv(1, 64, mid_channels=16)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, inp: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # version 1
        # x1 = self.inc(inp)

        # version 2
        x1 = self.inc(inp) + self.inc_mask(mask)

        # version 3
        # x = torch.cat([inp, mask], dim=1)
        # x1 = self.inc(x)

        # version 4
        # x1, mask = self.inc(inp), self.inc_mask(mask)
        # x1 = torch.cat([x1, mask], dim=1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # TODO: op = x + inp
        return x

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)
