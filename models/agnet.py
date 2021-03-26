import torch.nn as nn

from models.bottleneck_residual import BottleneckResidualBlock


class AGNet(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes=2):
        super(AGNet, self).__init__()
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes

        # self.conv_age = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 2, 1),
        #     BottleneckResidualBlock(16, 24, 2, 6),
        #     BottleneckResidualBlock(24, 24, 2, 6),
        #     BottleneckResidualBlock(24, 32, 2, 6),
        #     BottleneckResidualBlock(32, 32, 2, 6),
        #     BottleneckResidualBlock(32, 64, 2, 6),
        #     BottleneckResidualBlock(64, 64, 2, 6),
        #     BottleneckResidualBlock(64, 96, 1, 6),
        #     BottleneckResidualBlock(96, 96, 2, 6),
        #     BottleneckResidualBlock(96, 160, 2, 6),
        #     BottleneckResidualBlock(160, 160, 2, 6),
        #     BottleneckResidualBlock(160, 320, 1, 6),
        #     BottleneckResidualBlock(320, 320, 2, 6)
        # )
        # self.conv_gender = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 2, 1),
        #     BottleneckResidualBlock(16, 24, 2, 6),
        #     BottleneckResidualBlock(24, 32, 2, 6),
        #     BottleneckResidualBlock(32, 64, 2, 6),
        #     BottleneckResidualBlock(64, 96, 2, 6),
        #     BottleneckResidualBlock(96, 160, 2, 6),
        #     BottleneckResidualBlock(160, 320, 1, 6),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            BottleneckResidualBlock(32, 64, 2, 6),
            BottleneckResidualBlock(64, 64, 1, 6),
            BottleneckResidualBlock(64, 64, 1, 6)
        )
        self.conv3 = nn.Sequential(
            BottleneckResidualBlock(64, 96, 2, 6),
            BottleneckResidualBlock(96, 96, 1, 6),
            BottleneckResidualBlock(96, 96, 1, 6)
        )
        self.conv4 = nn.Sequential(
            BottleneckResidualBlock(96, 160, 2, 6),
            BottleneckResidualBlock(160, 160, 1, 6),
            BottleneckResidualBlock(160, 160, 1, 6)
        )
        self.conv5 = nn.Sequential(
            BottleneckResidualBlock(160, 320, 2, 6),
            BottleneckResidualBlock(320, 320, 1, 6),
            BottleneckResidualBlock(320, 320, 1, 6)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0),
            # nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.classifier_age = nn.Sequential(
            nn.Linear(1280, 160),
            # nn.BatchNorm1d(160),
            nn.ReLU6(inplace=True),
            nn.Linear(160, self.num_age_classes)
        )
        self.classifier_gender = nn.Sequential(
            nn.Linear(1280, 160),
            nn.ReLU6(inplace=True),
            nn.Linear(160, self.num_gender_classes)
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
        # age, gender = self.conv_age(x), self.conv_gender(x)
        # age, gender = self.avgpool(age), self.avgpool(gender)
        # age, gender = age.reshape(age.shape[0], -1), gender.reshape(gender.shape[0], -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        age, gender = x, x

        age, gender = self.classifier_age(age), self.classifier_gender(gender)
        age, gender = age.squeeze(-1).squeeze(-1), gender.squeeze(-1).squeeze(-1)
        age, gender = self.softmax(age), self.softmax(gender)

        return age, gender


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    model = AGNet(26).cuda()
    summary(model, (3, 224, 224))
    dummy = torch.zeros(2, 3, 224, 224).cuda()
    print(model(dummy).shape)






















