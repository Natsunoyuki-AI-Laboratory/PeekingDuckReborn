from torch import nn
import torch

from peekingduck.pipeline.nodes.model.copernicus_fmv1.copernicus_fm_files.conv_module import ConvModule


class FCNHead(nn.Module):
    """
    Native PyTorch replacement for mmseg.models.decode_heads.FCNHead
    """
    def __init__(self, in_channels: int, channels: int, num_classes: int, 
                 num_convs: int = 2, concat_input: bool = True):
        super().__init__()
        self.concat_input = concat_input
        
        # Build consecutive ConvModules
        convs = []
        convs.append(ConvModule(in_channels, channels, kernel_size=3, padding=1))
        for _ in range(num_convs - 1):
            convs.append(ConvModule(channels, channels, kernel_size=3, padding=1))
        self.convs = nn.Sequential(*convs)
        
        # Determine input channels for the final classification layer
        conv_out_channels = channels + in_channels if concat_input else channels
        self.conv_seg = nn.Conv2d(conv_out_channels, num_classes, kernel_size=1)

    def forward(self, inputs: list) -> torch.Tensor:
        # Decode heads typically take a list of features and process the target scale
        x = inputs[-1] if isinstance(inputs, (list, tuple)) else inputs
        
        feat = self.convs(x)
        if self.concat_input:
            feat = torch.cat([x, feat], dim=1)
            
        return self.conv_seg(feat)
    