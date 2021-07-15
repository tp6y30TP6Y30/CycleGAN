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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = None, padding_mode = 'reflect', norm = True, attention = False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding if padding else kernel_size // 2, padding_mode = padding_mode)
        self.norm = nn.InstanceNorm2d(out_channels) if norm else nn.Identity()
        self.acti = nn.ReLU(True)
        self.attn = SEModule(out_channels, 4) if attention else None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        if self.attn:
            x = residual + self.attn(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding = None, padding_mode = 'reflect', attention = False):
        super(ResidualBlock, self).__init__()
        self.convs = nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size, stride, padding = padding if padding else kernel_size // 2, padding_mode = padding_mode),
                        nn.ReLU(True),
                        nn.Conv2d(channels, channels, kernel_size, stride, padding = padding if padding else kernel_size // 2, padding_mode = padding_mode),
                     )
        self.acti = nn.ReLU(True)

    def forward(self, x):
        residual = x
        x = self.convs(x) + residual
        x = self.acti(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(
                        ConvBlock(3, 64, 7, 1),
                        ConvBlock(64, 128, 3, 2),
                        ConvBlock(128, 256, 3, 2),
                        ResidualBlock(256, 3, 1),
                        ResidualBlock(256, 3, 1),
                        ResidualBlock(256, 3, 1),
                        ResidualBlock(256, 3, 1),
                        ResidualBlock(256, 3, 1),
                     )

    def forward(self, x):
        feature = self.convs(x)
        return feature

class TransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = None, output_padding = 0):
        super(TransBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding = padding if padding else kernel_size // 2, output_padding = output_padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.acti = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convs = nn.Sequential(
                            ResidualBlock(256, 3, 1),
                            ResidualBlock(256, 3, 1),
                            ResidualBlock(256, 3, 1),
                            ResidualBlock(256, 3, 1),
                       )
        self.deconvs = nn.Sequential(
                            self.get_upsample_block(256, 128, 3, 2, None, 1),
                            self.get_upsample_block(128, 64, 3, 2, None, 1),
                            ConvBlock(64, 3, 7, 1)
                       )

    def get_upsample_block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        upsample_block = TransBlock(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        return upsample_block

    def forward(self, x):
        x = self.convs(x)
        x = self.deconvs(x)
        return x

class GANetwork(nn.Module):
    def __init__(self):
        super(GANetwork, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        weights_init(self)
        print('GAN params: ', self.get_params())

    def forward(self, input):
        feature = self.encoder(input)
        fake_img = self.decoder(feature)
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
