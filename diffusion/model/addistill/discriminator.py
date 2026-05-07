import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, color_dim, discriminator_feature_dim):
        super(Discriminator, self).__init__()
        
        # Define the layers in the network
        self.model = nn.Sequential(
            nn.Conv2d(color_dim, discriminator_feature_dim, 3, 1, 1, bias=False),
            nn.SiLU(True),
            nn.Conv2d(
                discriminator_feature_dim,
                discriminator_feature_dim * 2,
                3,
                1,
                1,
                bias=False,
            ),
            nn.GroupNorm(4, discriminator_feature_dim * 2),
            nn.SiLU(True),
            nn.Conv2d(
                discriminator_feature_dim * 2,
                discriminator_feature_dim * 4,
                3,
                1,
                1,
                bias=False,
            ),
            nn.GroupNorm(4, discriminator_feature_dim * 4),
            nn.SiLU(True),
            nn.Conv2d(
                discriminator_feature_dim * 4,
                discriminator_feature_dim * 8,
                3,
                1,
                1,
                bias=False,
            ),
            nn.GroupNorm(4, discriminator_feature_dim * 8),
            nn.SiLU(True),
            nn.Conv2d(
                discriminator_feature_dim * 8,
                discriminator_feature_dim * 16,
                3,
                1,
                1,
                bias=False,
            ),
            nn.GroupNorm(4, discriminator_feature_dim * 16),
            nn.SiLU(True),
            nn.Conv2d(discriminator_feature_dim * 16, 1, 3, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        original_ndim = x.dim()
        if original_ndim == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        out = self.model(x)
        if original_ndim == 5:
            out = out.view(B, T, -1).mean(dim=1)
        return out

# Example usage:
# color_dim = 4  # For RGB input images
# discriminator_feature_dim = 64  # Example feature dimension
# model = Discriminator(color_dim, discriminator_feature_dim)
# print(model)
