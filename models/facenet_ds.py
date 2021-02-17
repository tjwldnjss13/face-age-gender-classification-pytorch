import torch
import torch.nn as nn

from models.dsconv import DSConv


class FaceNetDS(nn.Module):
    def __init__(self, num_age_classes):
        super(FaceNetDS, self).__init__()
        self.num_age_classes = num_age_classes
        self.conv1 = DSConv(3, 128, 3, 2, 1, activation=True, batch_norm=False)
        self.conv2_1 = DSConv(128, 256, 3, 2, 1, activation=True, batch_norm=False)
        self.conv2_2 = DSConv(256, 128, 1, 1, 0, activation=True, batch_norm=False)
        self.conv2_3 = DSConv(128, 256, 3, 1, 1, activation=False, batch_norm=False)
        self.conv2_skip = DSConv(128, 256, 3, 2, 1)
        self.conv3_1 = DSConv(256, 512, 3, 2, 1, activation=True, batch_norm=False)
        self.conv3_2 = DSConv(512, 256, 1, 1, 0, activation=True, batch_norm=False)
        self.conv3_3 = DSConv(256, 512, 3, 1, 1, activation=False, batch_norm=False)
        self.conv3_skip = DSConv(256, 512, 3, 2, 1)
        self.conv4_1 = DSConv(512, 1024, 3, 2, 1, activation=True, batch_norm=False)
        self.conv4_2 = DSConv(1024, 512, 1, 1, 0, activation=True, batch_norm=False)
        self.conv4_3 = DSConv(512, 1024, 3, 1, 1, activation=False, batch_norm=False)
        self.conv4_skip = DSConv(512, 1024, 3, 2, 1)
        # self.conv5_1 = DSConv(256, 512, 3, 1, 1, activation=True, batch_norm=False)
        # self.conv5_2 = DSConv(512, 256, 1, 1, 0, activation=True, batch_norm=False)
        # self.conv5_3 = DSConv(256, 512, 3, 1, 1, activation=False, batch_norm=False)
        # self.conv5_skip = DSConv(256, 512, 1, 1, 0)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gender = DSConv(1024, 2, 1, 1, 0, activation=True, batch_norm=True)
        self.fc_age = DSConv(1024, self.num_age_classes, 1, 1, 0, activation=True, batch_norm=False)

        self.fc = DSConv(1024, self.num_age_classes + 2, 1, 1, 0, activation=True, batch_norm=False)

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)

        skip = x
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.relu(x + self.conv2_skip(skip))

        skip = x
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.relu(x + self.conv3_skip(skip))

        skip = x
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.relu(x + self.conv4_skip(skip))

        x = self.adaptivepool(x)

        x = self.fc(x)
        x = x.squeeze()

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = FaceNetDS(61).cuda()
    summary(model, (3, 112, 112))