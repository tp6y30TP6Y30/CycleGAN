import torch
import torch.nn as nn

class SpatialPyramidPooling(nn.Module):
    def __init__(self, channels, pool_sizes = [3, 5, 9]):
        super(SpatialPyramidPooling, self).__init__()

        self.head_conv = nn.Sequential(
                            Conv_block(channels, channels // 2, 1, 1, 0),
                            Conv_block(channels // 2, channels, 3, 1, 0),
                            Conv_block(channels, channels // 2, 1, 1, 0),
                         )
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x] + features, dim = 1)

        return features

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation = 'leakyrelu', downsample = False, attention = False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding if padding else kernel_size // 2)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.acti = nn.ReLU(True) if activation == 'relu' else nn.LeakyReLU(0.2, True)
        self.downsample = nn.MaxPool2d(2) if downsample else nn.Identity()
        self.attention = attention
        if self.attention:
            self.attn = SEModule(out_channels, 4)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        if self.attention:
            x = residual + self.attn(x)
        x = self.downsample(x)
        return x

class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([
                        ConvBlock(channels[i], channels[i + 1], 3, 1, 1, downsample = True, attention = channels[i] == channels[i + 1])
                        for i in range(len(channels) - 1)
                     ])

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        return features

class TransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
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
                            ConvBlock(512, 256, 1, 1, 0, activation = 'relu'),
                            ConvBlock(512, 256, 1, 1, 0, activation = 'relu'),
                            ConvBlock(256, 128, 1, 1, 0, activation = 'relu'),
                            ConvBlock(256, 128, 1, 1, 0, activation = 'relu')
                        ])

    def get_upsample_block(self, in_channels, out_channels):
        if self.up_mode == 'up_sample':
            upsample_block = nn.Sequential(
                                nn.Upsample(scale_factor = 2),
                                ConvBlock(in_channels, out_channels, 3, 1, 1, activation = 'relu', downsample = False, attention = in_channels == out_channels)
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
    def __init__(self, in_channels = 3):
        super(GANetwork, self).__init__()
        channels = [64, 128, 128, 256, 256, 512]
        self.stem_conv = ConvBlock(in_channels, channels[0], 3, 1, 1)
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels[::-1])
        self.leaf_conv = nn.Conv2d(channels[::-1][-1], in_channels, 3, 1, 1)
        weights_init(self)
        print('GAN params: ', self.get_params())

    def forward(self, input):
        x = self.stem_conv(input)
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        fake_img = self.leaf_conv(features)
        assert(input.shape == fake_img.shape)
        return fake_img

    def get_params(self):
        return sum(p.numel() for p in self.parameters())

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('InstancehNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    input = torch.rand([2, 3, 256, 256])
    model = GANetwork()
    print(model)
    print(model(input).shape)
    print(model.get_params())
