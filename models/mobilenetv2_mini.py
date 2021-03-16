import torch.nn as nn

from models.bottleneck_residual import BottleneckResidualBlock
from torchvision.models import resnet50, mobilenet_v2


class MobilenetV2Mini(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes=2):
        super(MobilenetV2Mini, self).__init__()
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes

        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Sequential(
            BottleneckResidualBlock(16, 24, 2, 6),
            BottleneckResidualBlock(24, 24, 2, 6)
        )
        self.conv3 = nn.Sequential(
            BottleneckResidualBlock(24, 32, 2, 6),
            BottleneckResidualBlock(32, 32, 2, 6)
        )
        self.conv4 = nn.Sequential(
            BottleneckResidualBlock(32, 64, 2, 6),
            BottleneckResidualBlock(64, 64, 2, 6)
        )
        self.conv5 = nn.Sequential(
            BottleneckResidualBlock(64, 96, 1, 6),
            BottleneckResidualBlock(96, 96, 2, 6)
        )
        self.conv6 = nn.Sequential(
            BottleneckResidualBlock(96, 160, 2, 6),
            BottleneckResidualBlock(160, 160, 2, 6)
        )
        # self.conv7 = nn.Sequential(
        #     BottleneckResidualBlock(160, 320, 1, 6),
        #     BottleneckResidualBlock(320, 320, 2, 6)
        # )
        self.conv7 = nn.Sequential(
            # nn.Conv2d(320, 1280, 1, 1, 0),
            # nn.BatchNorm2d(1280),
            nn.Conv2d(160, 640, 1, 1, 0),
            nn.BatchNorm2d(640),
            nn.LeakyReLU(inplace=True),
        )
        self.classifier_age = nn.Sequential(
            nn.Linear(640, 160),
            nn.BatchNorm1d(160),
            nn.ReLU6(inplace=True),
            nn.Dropout(),
            nn.Linear(160, self.num_age_classes)
        )
        self.classifier_gender = nn.Sequential(
            nn.Linear(640, 80),
            # nn.BatchNorm1d(80),
            nn.ReLU6(inplace=True),
            # nn.Dropout(),
            nn.Linear(80, self.num_gender_classes)
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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # x = self.conv8(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        age, gender = self.classifier_age(x), self.classifier_gender(x)
        age, gender = age.squeeze(-1).squeeze(-1), gender.squeeze(-1).squeeze(-1)
        age, gender = self.softmax(age), self.softmax(gender)

        return age, gender


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    model = MobilenetV2Mini(26).cuda()
    summary(model, (3, 224, 224))
    dummy = torch.zeros(2, 3, 224, 224).cuda()
    print(model(dummy).shape)






















