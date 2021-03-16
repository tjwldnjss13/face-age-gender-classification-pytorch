import torch.nn as nn

from models.bottleneck_residual import BottleneckResidualBlock
from torchvision.models import resnet50, mobilenet_v2


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
        self.conv9 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0),
            nn.ReLU6(True),
        )
        self.dense = nn.Sequential(
            nn.Linear(1280, 320),
            nn.ReLU6(True),
        )
        self.classifier = nn.Linear(320, num_age_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        # nn.init.kaiming_normal_(self.conv9.weight, mode='fan_out')
        # nn.init.kaiming_normal_(self.conv10.weight, mode='fan_out')

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
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        x = self.classifier(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    model = MobilenetV2(11).cuda()
    summary(model, (3, 224, 224))
    dummy = torch.zeros(2, 3, 224, 224).cuda()
    print(model(dummy).shape)






















