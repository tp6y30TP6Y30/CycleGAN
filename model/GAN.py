import torch
import torch.nn as nn

class SEModule(nn.Module):
    def __init__(self, channels, reduction=1):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x
        x = self.avg_pool(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return original * x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, activation = 'leakyrelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.acti = nn.ReLU(True) if activation == 'relu' else nn.LeakyReLU(0.2, True)
        self.redu = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.attn = SEModule(out_channels, 4)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        x = self.attn(x)
        residual = self.redu(residual)
        x = residual + x
        return x

class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([
                        ConvBlock(channels[i], channels[i + 1])
                        for i in range(len(channels) - 1)
                     ])

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        return features

class TransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1):
        super(TransBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.acti = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels, up_mode = 'up_sample'):
        super(Decoder, self).__init__()
        self.up_mode = up_mode
        self.deconvs = nn.ModuleList([
                            self.get_upsample_block(channels[i], channels[i + 1])
                            for i in range(len(channels) - 1)
                       ])
        self.cat_convs = nn.ModuleList([
                            ConvBlock(channels[i], channels[i] // 2, 1, 1, 0, activation = 'relu')
                            for i in range(len(channels) - 2)
                        ])

    def get_upsample_block(self, in_channels, out_channels):
        if self.up_mode == 'up_sample':
            upsample_block = nn.Sequential(
                                nn.Upsample(scale_factor = 2),
                                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                nn.InstanceNorm2d(out_channels),
                                nn.ReLU(True)
                             )
        elif self.up_mode == 'transpose':
            upsample_block = TransBlock(in_channels, out_channels)
        return upsample_block

    def forward(self, features):
        output = features[0]
        for i in range(len(self.deconvs)):
            output = self.deconvs[i](output)
            if i == len(self.deconvs) - 1: break
            output = torch.cat([output, features[i + 1]], dim = 1)
            output = self.cat_convs[i](output)
        return output

class GANetwork(nn.Module):
    def __init__(self, in_channels = 3, num_stages = 6):
        super(GANetwork, self).__init__()
        channels = [in_channels, *[64 * 2**stage for stage in range(num_stages)]]
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels[::-1])

    def forward(self, input):
        features = self.encoder(input)
        fake_img = self.decoder(features[::-1])
        assert(input.shape == fake_img.shape)
        return fake_img

    def get_params(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == '__main__':
    input = torch.rand([2, 3, 256, 256])
    model = GANetwork()
    print(model)
    print(model(input).shape)
    print(model.get_params())
