from torch import nn
import torch
import torch.nn.functional as F

from peekingduck.pipeline.nodes.model.copernicus_fmv1.copernicus_fm_files.conv_module import ConvModule


class PyramidPoolingModule(nn.Module):
    def __init__(self, pool_scales: tuple, in_channels: int, channels: int):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels, channels, kernel_size=1)
            ) for scale in pool_scales
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [x]
        for feat in self.features:
            # Pool, Conv, then interpolate back to original size
            pooled = feat(x)
            upsampled = F.interpolate(
                pooled, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            res.append(upsampled)
        return torch.cat(res, dim=1)


class UPerHead(nn.Module):
    """
    Native PyTorch replacement for mmseg.models.decode_heads.UPerHead
    """
    def __init__(self, in_channels_list: list, channels: int, num_classes: int, 
                 pool_scales: tuple = (1, 2, 3, 6)):
        super().__init__()
        
        # Pyramid Pooling Module on the highest level feature map
        self.psp_modules = PyramidPoolingModule(pool_scales, in_channels_list[-1], channels)
        
        # Bottleneck after PPM
        self.bottleneck = ConvModule(
            in_channels_list[-1] + len(pool_scales) * channels, 
            channels, kernel_size=3, padding=1
        )

        # FPN Lateral and Output Convs
        self.lateral_convs = nn.ModuleList([
            ConvModule(in_c, channels, kernel_size=1) 
            for in_c in in_channels_list[:-1]
        ])
        
        self.fpn_convs = nn.ModuleList([
            ConvModule(channels, channels, kernel_size=3, padding=1) 
            for _ in in_channels_list[:-1]
        ])
        
        # Final fusion and classification
        self.fpn_bottleneck = ConvModule(
            len(in_channels_list) * channels, channels, kernel_size=3, padding=1
        )
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs: list) -> torch.Tensor:
        # 1. Process lateral connections
        laterals = [
            lat_conv(inputs[i]) for i, lat_conv in enumerate(self.lateral_convs)
        ]
        
        # 2. Process top-level feature map with PPM
        psp_out = self.psp_modules(inputs[-1])
        laterals.append(self.bottleneck(psp_out))

        # 3. Top-down FPN pathway (add higher level features to lower level features)
        for i in range(len(laterals) - 1, 0, -1):
            target_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i], size=target_shape, mode='bilinear', align_corners=False
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 4. FPN Convs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)
        ]
        fpn_outs.append(laterals[-1])

        # 5. Concatenate all FPN levels to the highest resolution
        target_shape = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_shape, mode='bilinear', align_corners=False
            )
        fpn_out = torch.cat(fpn_outs, dim=1)

        # 6. Final classification
        out = self.fpn_bottleneck(fpn_out)
        out = self.conv_seg(out)
        return out
