import torch
import torch.nn as nn


class ParallelFusionAttention(nn.Module):
    """Enhanced TwinAttention with residual connection"""

    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super().__init__()
        # Channel attention
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention (efficient design)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.Conv2d(in_channels // reduction, in_channels // reduction,
                      kernel_size, padding=kernel_size // 2,
                      groups=in_channels // reduction, bias=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Fusion with residual connection
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # Parallel attention computation
        ca = self.channel_branch(x)
        sa = self.spatial_branch(x)

        # Apply attention
        channel_enhanced = x * ca
        spatial_enhanced = x * sa

        # Concatenate and fuse
        combined = torch.cat((channel_enhanced, spatial_enhanced), dim=1)
        fused = self.fuse(combined)

        # Residual connection
        return x + fused