# efficient_unet.py

import torch
import torch.nn as nn
import timm
from unet_parts import Up, OutConv

class UNetEfficientNet(nn.Module):
    def __init__(self, n_channels=10, n_classes=1, bilinear=True):
        super(UNetEfficientNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Create EfficientNet-B4 backbone with correct input channels
        self.encoder = timm.create_model(
            'efficientnet_b4',
            features_only=True,
            pretrained=True,
            in_chans=n_channels
        )

        # Fetch actual encoder feature channels: [24, 32, 56, 160, 448]
        enc_channels = self.encoder.feature_info.channels()
        assert enc_channels == [24, 32, 56, 160, 448], f"Unexpected encoder channels: {enc_channels}"

        # Determine decoder channel sizes
        factor = 2 if bilinear else 1
        dec1 = 256 // factor   # 128
        dec2 = 128 // factor   #  64
        dec3 =  64 // factor   #  32
        dec4 =  32             #  32

        # Build decoder, summing actual feature channels at each skip connection
        self.up1 = Up(enc_channels[4] + enc_channels[3], dec1, bilinear)  # 448 + 160 -> 608 → 128
        self.up2 = Up(dec1 +        enc_channels[2], dec2, bilinear)     # 128 +  56 -> 184 →  64
        self.up3 = Up(dec2 +        enc_channels[1], dec3, bilinear)     #  64 +  32 ->  96 →  32
        self.up4 = Up(dec3 +        enc_channels[0], dec4, bilinear)     #  32 +  24 ->  56 →  32

        self.outc = OutConv(dec4, n_classes)

    def forward(self, x):
        # Encoder: returns list of feature maps [x1, x2, x3, x4, x5]
        x1, x2, x3, x4, x5 = self.encoder(x)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)
