import torch.nn as nn

from models.bottleneck_residual import BottleneckResidualBlock


class AGNet(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes=2):
        super(AGNet, self).__init__()
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes

        # self.conv_age = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 2, 1),
        #     BottleneckResidualBlock(16, 32, 2, 6),
        #     BottleneckResidualBlock(32, 32, 1, 6),
        #     BottleneckResidualBlock(32, 64, 2, 6),
        #     BottleneckResidualBlock(64, 64, 1, 6),
        #     BottleneckResidualBlock(64, 96, 1, 6),
        #     BottleneckResidualBlock(96, 96, 1, 6),
        #     BottleneckResidualBlock(96, 160, 2, 6),
        #     BottleneckResidualBlock(160, 160, 1, 6),
        #     BottleneckResidualBlock(160, 320, 2, 6),
        #     BottleneckResidualBlock(320, 320, 1, 6),
        #     nn.Conv2d(320, 1280, 1, 1, 0),
        #     nn.ReLU6(inplace=True)
        # )
        # self.conv_gender = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 2, 1),
        #     BottleneckResidualBlock(16, 32, 2, 6),
        #     BottleneckResidualBlock(32, 32, 1, 6),
        #     BottleneckResidualBlock(32, 32, 1, 6),
        #     BottleneckResidualBlock(32, 64, 2, 6),
        #     BottleneckResidualBlock(64, 64, 1, 6),
        #     BottleneckResidualBlock(64, 64, 1, 6),
        #     BottleneckResidualBlock(64, 96, 1, 6),
        #     BottleneckResidualBlock(96, 96, 1, 6),
        #     BottleneckResidualBlock(96, 96, 1, 6),
        #     BottleneckResidualBlock(96, 160, 2, 6),
        #     BottleneckResidualBlock(160, 160, 1, 6),
        #     BottleneckResidualBlock(160, 160, 1, 6),
        #     BottleneckResidualBlock(160, 320, 2, 6),
        #     BottleneckResidualBlock(320, 320, 1, 6),
        #     BottleneckResidualBlock(320, 320, 1, 6),
        #     nn.Conv2d(320, 1280, 1, 1, 0),
        #     nn.ReLU6(inplace=True)
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2_age = nn.Sequential(
            BottleneckResidualBlock(8, 16, 2, 6),
        )
        self.conv3_age = nn.Sequential(
            BottleneckResidualBlock(16, 32, 2, 6),
        )
        # self.conv4_age = nn.Sequential(
        #     BottleneckResidualBlock(32, 64, 2, 6),
        # )
        # self.conv5_age = nn.Sequential(
        #     BottleneckResidualBlock(64, 128, 1, 6),
        # )
        # self.conv6_age = nn.Sequential(
        #     BottleneckResidualBlock(96, 160, 2, 6),
        # )
        # self.conv7_age = nn.Sequential(
        #     BottleneckResidualBlock(160, 320, 1, 6)
        # )
        # self.conv5_age = nn.Sequential(
        #     nn.Conv2d(64, 128, 1, 1, 0),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(inplace=True)
        # )
        self.conv2_gender = nn.Sequential(
            BottleneckResidualBlock(8, 16, 2, 6),
        )
        self.conv3_gender = nn.Sequential(
            BottleneckResidualBlock(16, 32, 2, 6),
        )
        self.conv4_gender = nn.Sequential(
            BottleneckResidualBlock(32, 64, 2, 6),
        )
        # self.conv5_gender = nn.Sequential(
        #     BottleneckResidualBlock(96, 160, 2, 6),
        #     BottleneckResidualBlock(160, 160, 1, 6),
        # )
        # self.conv6_gender = nn.Sequential(
        #     BottleneckResidualBlock(160, 320, 2, 6),
        #     BottleneckResidualBlock(320, 320, 1, 6),
        # )
        self.conv5_gender = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.classifier_age = nn.Sequential(
            # nn.Conv2d(128, self.num_age_classes, 1, 1, 0)
            nn.Linear(32 * 14 * 14, self.num_age_classes)
        )
        self.classifier_gender = nn.Sequential(
            nn.Conv2d(128, self.num_gender_classes, 1, 1, 0)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout()

        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        age = gender = self.conv1(x)

        age = self.conv2_age(age)
        age = self.conv3_age(age)

        gender = self.conv2_gender(gender)
        gender = self.conv3_gender(gender)
        gender = self.conv4_gender(gender)
        gender = self.conv5_gender(gender)

        age = age.reshape(age.shape[0], -1)

        gender = self.avgpool(gender)

        age = self.classifier_age(age)
        gender = self.classifier_gender(gender)
        age, gender = age.squeeze(), gender.squeeze()
        age, gender = self.softmax(age), self.softmax(gender)

        return age, gender


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    model = AGNet(26).cuda()
    summary(model, (3, 112, 112))
    dummy = torch.zeros(2, 3, 112, 112).cuda()
    age, gen = model(dummy)
    print(age.shape)
    print(gen.shape)






















