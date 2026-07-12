from torch import nn


class Feature2Pyramid(nn.Module):
    """
    Native PyTorch replacement for mmseg.models.necks.Feature2Pyramid
    Maps backbone features to a feature pyramid.
    """
    def __init__(self, embed_dim: int, rescales: list = [4, 2, 1, 0.5]):
        super().__init__()
        self.rescales = rescales
        
        # Create a projection layer for each requested scale
        self.proj_layers = nn.ModuleList()
        for scale in rescales:
            if scale == 4:
                self.proj_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
                ))
            elif scale == 2:
                self.proj_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
                ))
            elif scale == 1:
                self.proj_layers.append(nn.Identity())
            elif scale == 0.5:
                self.proj_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"Scale {scale} not supported in Feature2Pyramid rewrite.")

    def forward(self, inputs: list) -> tuple:
        # Assuming inputs is a list of features from the backbone. 
        # Often for ViTs, it's just one feature map repeated, or a list of intermediate maps.
        if not isinstance(inputs, list):
            inputs = [inputs] * len(self.rescales)
            
        assert len(inputs) == len(self.proj_layers)
        
        outs = []
        for i, proj in enumerate(self.proj_layers):
            outs.append(proj(inputs[i]))
            
        return tuple(outs)
