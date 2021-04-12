import torch.nn as nn

from models.bottleneck_residual import BottleneckResidualBlock


class AgeNet(nn.Module):
    def __init__(self, num_age_classes):
        super(AgeNet, self).__init__()
        self.num_age_classes = num_age_classes

        self.conv_age = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            BottleneckResidualBlock(16, 32, 2, 6),
            BottleneckResidualBlock(32, 32, 1, 6),
            BottleneckResidualBlock(32, 64, 2, 6),
            BottleneckResidualBlock(64, 64, 1, 6),
            BottleneckResidualBlock(64, 96, 1, 6),
            BottleneckResidualBlock(96, 96, 1, 6),
            BottleneckResidualBlock(96, 160, 2, 6),
            BottleneckResidualBlock(160, 160, 1, 6),
            BottleneckResidualBlock(160, 320, 2, 6),
            BottleneckResidualBlock(320, 320, 1, 6),
            nn.Conv2d(320, 1280, 1, 1, 0),
            nn.ReLU6(inplace=True)
        )
        self.classifier_age = nn.Sequential(
            nn.Conv2d(1280, 640, 1, 1, 0),
            nn.ReLU6(inplace=True),
            nn.Conv2d(640, 160, 1, 1, 0),
            nn.ReLU6(inplace=True),
            nn.Conv2d(160, self.num_age_classes, 1, 1, 0)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout()

        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_age(x)
        x = self.avgpool(x)
        x = self.classifier_age(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    model = AgeNet(26).cuda()
    summary(model, (3, 224, 224))
    dummy = torch.zeros(2, 3, 224, 224).cuda()
    print(model(dummy).shape)






















