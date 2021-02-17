import torch
import torch.nn as nn

from models.dsconv import DSConv


class FaceNetDS(nn.Module):
    def __init__(self, num_age_classes):
        super(FaceNetDS, self).__init__()
        self.num_age_classes = num_age_classes
        self.conv1 = DSConv(3, 32, 3, 1, 1, activation=True, batch_norm=False)
        self.conv2_1 = DSConv(32, 64, 3, 1, 1, activation=True, batch_norm=False)
        self.conv2_2 = DSConv(64, 32, 1, 1, 0, activation=True, batch_norm=False)
        self.conv2_3 = DSConv(32, 64, 3, 1, 1, activation=True, batch_norm=False)
        self.conv3_1 = DSConv(64, 128, 3, 1, 1, activation=True, batch_norm=False)
        self.conv3_2 = DSConv(128, 64, 1, 1, 0, activation=True, batch_norm=False)
        self.conv3_3 = DSConv(64, 128, 3, 1, 1, activation=True, batch_norm=False)
        self.conv4_1 = DSConv(128, 256, 3, 1, 1, activation=True, batch_norm=False)
        self.conv4_2 = DSConv(256, 128, 1, 1, 0, activation=True, batch_norm=False)
        self.conv4_3 = DSConv(128, 256, 3, 1, 1, activation=True, batch_norm=False)
        self.conv5_1 = DSConv(256, 512, 3, 1, 1, activation=True, batch_norm=False)
        self.conv5_2 = DSConv(512, 256, 1, 1, 0, activation=True, batch_norm=False)
        self.conv5_3 = DSConv(256, 512, 3, 1, 1, activation=True, batch_norm=False)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gender1 = DSConv(512, 64, 1, 1, 0, activation=True, batch_norm=False)
        self.fc_gender2 = DSConv(64, 2, 1, 1, 0, activation=True, batch_norm=False)
        self.fc_age1 = DSConv(512, 256, 1, 1, 0, activation=True, batch_norm=False)
        self.fc_age2 = DSConv(256, self.num_age_classes, 1, 1, 0, activation=True, batch_norm=False)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.adaptivepool(x)

        gender = self.fc_gender1(x)
        gender = self.dropout(gender)
        gender = self.fc_gender2(gender)

        age = self.fc_age1(x)
        age = self.dropout(age)
        age = self.fc_age2(age)

        gender = gender.squeeze()
        age = age.squeeze()

        gender = self.softmax(gender)
        age = self.softmax(age)

        return age, gender


if __name__ == '__main__':
    from torchsummary import summary
    model = FaceNetDS(61).cuda()
    summary(model, (3, 112, 112))