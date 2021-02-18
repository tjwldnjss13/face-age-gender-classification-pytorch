import torch.nn as nn

from models.bottleneck_residual import BottleneckResidualBlock


class MobilenetV2(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes=2):
        super(MobilenetV2, self).__init__()
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = BottleneckResidualBlock(32, 16, 1, 1)
        self.conv3 = nn.Sequential(
            BottleneckResidualBlock(16, 24, 2, 6),
            BottleneckResidualBlock(24, 24, 1, 6),
        )
        self.conv4 = nn.Sequential(
            BottleneckResidualBlock(24, 32, 2, 6),
            BottleneckResidualBlock(32, 32, 1, 6),
            BottleneckResidualBlock(32, 32, 1, 6),
        )
        self.conv5 = nn.Sequential(
            BottleneckResidualBlock(32, 64, 2, 6),
            BottleneckResidualBlock(64, 64, 1, 6),
            BottleneckResidualBlock(64, 64, 1, 6),
            BottleneckResidualBlock(64, 64, 1, 6),
        )
        self.conv6 = nn.Sequential(
            BottleneckResidualBlock(64, 96, 1, 6),
            BottleneckResidualBlock(96, 96, 1, 6),
            BottleneckResidualBlock(96, 96, 1, 6),
        )
        self.conv7 = nn.Sequential(
            BottleneckResidualBlock(96, 160, 2, 6),
            BottleneckResidualBlock(160, 160, 1, 6),
            BottleneckResidualBlock(160, 160, 1, 6),
        )
        self.conv8 = BottleneckResidualBlock(160, 320, 1, 6)
        self.conv9 = nn.Conv2d(320, 1280, 1, 1, 0)
        self.conv10 = nn.Conv2d(1280, num_age_classes + num_gender_classes, 1, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.kaiming_normal_(self.conv10.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.avgpool(x)
        x = self.conv10(x)

        x = x.squeeze()

        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    model = MobilenetV2(61).cuda()
    summary(model, (3, 224, 224))
    dummy = torch.zeros(2, 3, 224, 224).cuda()
    print(model(dummy).shape)






















