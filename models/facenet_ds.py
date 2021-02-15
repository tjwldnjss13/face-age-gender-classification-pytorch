import torch
import torch.nn as nn

from models.dsconv import DSConv


class FaceNetDS(nn.Module):
    def __init__(self, num_age_classes):
        super(FaceNetDS, self).__init__()
        self.num_age_classes = num_age_classes
        self.conv1 = DSConv(3, 128, 3, 1, 1, True)
        self.conv2 = DSConv(128, 256, 3, 1, 1, True)
        self.conv3 = DSConv(256, 512, 3, 1, 1, True)
        self.conv4 = DSConv(512, 1024, 3, 1, 1, True)
        self.fc_gender1 = DSConv(1024, 64, 7, 1, 0, True)
        self.fc_gender2 = DSConv(64, 2, 1, 1, 0, True)
        self.fc_age = DSConv(1024, self.num_age_classes, 7, 1, 0, True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.maxpool(x)

        gender = self.fc_gender1(x)
        gender = self.fc_gender2(gender)

        age = self.fc_age(x)

        gender = gender.squeeze()
        age = age.squeeze()

        gender = self.softmax(gender)
        age = self.softmax(age)

        return age, gender


if __name__ == '__main__':
    from torchsummary import summary
    model = FaceNetDS(20).cuda()
    summary(model, (3, 112, 112))