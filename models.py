import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BasicBlock(nn.Module):
    """A lightweight ResNet-style basic block."""
    expansion = 1
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.act   = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.act(out + identity)
        return out

class ConvEncoder(nn.Module):
    """
    CNN backbone adapted for 64x64 images:
      stem -> 4 stages with downsampling -> global average pool -> linear to feature_dim
    Output: (B, feature_dim)
    """
    def __init__(self, input_channels: int, feature_dim: int):
        super().__init__()
        width = 64  # base width
        self.stem = ConvBNReLU(input_channels, width, k=3, s=1, p=1)  # keep resolution at 64x64

        # 64x64
        self.layer1 = self._make_layer(width,   width,  blocks=2, stride=1)  # 64x64
        # 32x32
        self.layer2 = self._make_layer(width,   width*2, blocks=2, stride=2) # 32x32
        # 16x16
        self.layer3 = self._make_layer(width*2, width*4, blocks=2, stride=2) # 16x16
        # 8x8
        self.layer4 = self._make_layer(width*4, width*8, blocks=2, stride=2) # 8x8

        self.gap = nn.AdaptiveAvgPool2d(1)              # -> (B, 512, 1, 1)
        self.fc  = nn.Linear(width*8, feature_dim)       # 512 -> 1000

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)         # 64x64
        x = self.layer1(x)       # 64x64
        x = self.layer2(x)       # 32x32
        x = self.layer3(x)       # 16x16
        x = self.layer4(x)       # 8x8
        x = self.gap(x)          # (B, C, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.fc(x)           # (B, feature_dim)
        return x


class ImageEncoder(nn.Module):
    """
    A simple neural network template for self-supervised learning.

    Structure:
    1. Encoder: Maps an input image of shape 
       (input_channels, input_dim, input_dim) 
       into a lower-dimensional feature representation.
    2. Projector: Transforms the encoder output into the final 
       embedding space of size `proj_dim`.

    Notes:
    - DO NOT modify the fixed class variables: 
      `input_dim`, `input_channels`, and `feature_dim`.
    - You may freely modify the architecture of the encoder 
      and projector (layers, activations, normalization, etc.).
    - You may add additional helper functions or class variables if needed.
    """

    ####### DO NOT MODIFY THE CLASS VARIABLES #######
    input_dim: int = 64
    input_channels: int = 3
    feature_dim: int = 1000
    proj_dim: int = 128
    #################################################

    def __init__(self):
        super().__init__()

        # Encoder: convolutional backbone producing (B, feature_dim=1000)
        self.encoder = ConvEncoder(self.input_channels, self.feature_dim)

        # Projector (SimCLR-style MLP): 1000 -> 2048 -> 128
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.proj_dim, bias=True),
        )

        # initialize projector last layer to small weights to help for stability
        nn.init.trunc_normal_(self.projector[-1].weight, std=0.02)
        if getattr(self.projector[-1], "bias", None) is not None:
            nn.init.zeros_(self.projector[-1].bias)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, proj_dim).
        """
        features = self.encoder(x)                    # (B, 1000)
        projected_features = self.projector(features) # (B, 128)
        return features, projected_features
    
    def get_features(self, x):
        """
        Get the features from the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output features of shape (batch_size, feature_dim).
        """
        features = self.encoder(x)
        return features