import torch
import torch.nn as nn

architecture_config = [
    (32, 3, 1), #(out_channel, kernel_size, stride)
    (64, 3, 2),
    ["B", 1], # residual block , #repeat
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, b_n = True,**kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=not b_n, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.b_n = b_n

    def forward(self, x):
        convolution = self.conv(x)
        if self.b_n:
            batch_norm = self.batch_norm(convolution)
            convolution = self.activation(batch_norm)
        return convolution

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, use_res = True, num_repeat = 1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeat):
            self.layers +=[
                nn.Sequential(
                CNNBlock(in_channels, in_channels//2, kernel_size = 1),
                CNNBlock(in_channels//2, in_channels, kernel_size = 3,padding = 1)
                )
            ]
        self.use_res = use_res
        self.num_repeat = num_repeat

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)+x if self.use_res else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes ):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size = 3, padding = 1),
            CNNBlock(2*in_channels, (num_classes + 5) * 3,b_n = False, kernel_size = 3, padding = 1)
        )
        self.num_classes = num_classes

    def forward(self,x):
        pred = self.pred(x)
        pred = pred.reshape(x.shape[0],3,self.num_classes+5, x.shape[2],x.shape[3])
        pred = pred.permute(0,1,3,4,2) #(N,3,grids,grids, 5+num_classes)
        return pred

class YOLOV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeat == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for x in architecture_config:
            if isinstance(x,tuple):
                out_channels, kernel_size, stride = x
                layers.append(
                    CNNBlock(in_channels, out_channels,kernel_size = kernel_size, stride = stride, padding = 1 if kernel_size==3 else 0)
                )
                in_channels = out_channels
            elif isinstance(x,list):
                _, num_repeat = x
                layers.append(
                    ResidualBlock(in_channels, num_repeat=num_repeat)
                )
            elif isinstance(x, str):
                if x == "S":
                    layers +=[
                        ResidualBlock(in_channels, use_res=False, num_repeat=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels//2
                elif x == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels*3
        return layers


