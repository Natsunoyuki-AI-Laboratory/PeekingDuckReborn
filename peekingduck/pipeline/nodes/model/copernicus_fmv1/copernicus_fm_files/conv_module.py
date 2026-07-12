import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """
    Native PyTorch replacement for mmcv.cnn.bricks.conv_module.ConvModule
    Defaults to Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, 
                 bias: bool = False, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        
        layers = []
        # 1. Convolution
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias=bias
        ))
        
        # 2. Normalization
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            
        # 3. Activation
        if act_layer is not None:
            layers.append(act_layer(inplace=True))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
