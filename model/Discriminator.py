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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        feature = self.avg_pool(x).squeeze()
        return feature

class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        hidden = in_channels // 2
        self.classify = nn.Sequential(
                            nn.Linear(in_channels, hidden),
                            nn.ReLU(True),
                            nn.Linear(hidden, out_channels)
                        )

    def forward(self, feature):
        return self.classify(feature)

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Discriminator, self).__init__()
        channels = [64, 128, 128, 256, 256, 512]
        self.stem_conv = ConvBlock(in_channels, channels[0], 3, 1, 1)
        self.encoder = Encoder(channels)
        self.classifier = Classifier(in_channels = channels[-1], out_channels = 2)
        weights_init(self)
        print('Discr params: ', self.get_params())

    def forward(self, input):
        x = self.stem_conv(input)
        feature = self.encoder(x)
        predict = self.classifier(feature)
        return predict

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
    model = Discriminator()
    print(model)
    print(model(input).shape)
    print(model.get_params())