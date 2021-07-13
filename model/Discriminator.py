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
    def __init__(self, in_channels = 3, num_stages = 5):
        super(Discriminator, self).__init__()
        channels = [in_channels, *[64 * 2**stage for stage in range(num_stages)]]
        self.encoder = Encoder(channels)
        self.classifier = Classifier(in_channels = 1024, out_channels = 1)

    def forward(self, x):
        feature = self.encoder(x)
        predict = self.classifier(feature)
        return predict

    def get_params(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == '__main__':
    input = torch.rand([2, 3, 256, 256])
    model = Discriminator()
    print(model)
    print(model(input).shape)
    print(model.get_params())