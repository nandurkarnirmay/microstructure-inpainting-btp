from __future__ import annotations

import torch
import torch.nn as nn


def _norm(ch: int) -> nn.Module:
    return nn.InstanceNorm2d(ch, affine=True)


class MaskGuidedGenerator(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 48):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 4, 2, 1),
            _norm(c2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 4, 2, 1),
            _norm(c3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(c3, c4, 4, 2, 1),
            _norm(c4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1),
            _norm(c4),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, 2, 1),
            _norm(c3),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(c3 * 2, c2, 4, 2, 1),
            _norm(c2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(c2 * 2, c1, 4, 2, 1),
            _norm(c1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c1 * 2, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked, mask], dim=1)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        m = self.mid(s4)
        d4 = self.dec4(m)
        d3 = self.dec3(torch.cat([d4, s3], dim=1))
        d2 = self.dec2(torch.cat([d3, s2], dim=1))
        out = self.dec1(torch.cat([d2, s1], dim=1))
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 64):
        super().__init__()
        def block(i: int, o: int, norm: bool = True):
            layers = [nn.Conv2d(i, o, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if norm:
                layers.insert(1, nn.InstanceNorm2d(o, affine=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_channels, base, norm=False),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 8, norm=False),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)
        return self.net(x)


class SpatialNoiseGenerator(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        z_channels: int = 1,
        base: int = 64,
        scale_factor: int = 8,
        num_upsample: int = 1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_upsample = num_upsample
        ch = base * (2 ** num_upsample)
        self.stem = nn.Sequential(
            nn.Conv2d(z_channels, ch, 3, 1, 1),
            _norm(ch),
            nn.ReLU(inplace=True),
        )
        ups = []
        c = ch
        for _ in range(num_upsample):
            next_c = max(base, c // 2)
            ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c, next_c, scale_factor, scale_factor, 0),
                    _norm(next_c),
                    nn.ReLU(inplace=True),
                )
            )
            c = next_c
        self.ups = nn.ModuleList(ups)
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            _norm(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
        x = self.stem(z)
        for up in self.ups:
            x = up(x)
        if out_hw is not None:
            x = nn.functional.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return self.refine(x)


def spatial_seed_size_for_hole(hole: int, scale_factor: int = 8) -> int:
    s = hole // scale_factor + 2
    return max(4, s)
