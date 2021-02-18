import torch.nn as nn

from models.dsconv import DSConv


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(BottleneckResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_factor = expansion_factor

        self.mid_channels = in_channels * expansion_factor
        self.conv1x1_1 = nn.Conv2d(in_channels, self.mid_channels, 1, 1, 0)
        self.dsconv = DSConv(self.mid_channels, self.mid_channels, 3, stride, 1, False, False)
        self.conv1X1_2 = nn.Conv2d(self.mid_channels, out_channels, 1, 1, 0)
        self.relu6 = nn.ReLU6(True)
        self.lrelu = nn.LeakyReLU(.1, True)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.kaiming_normal_(self.conv1x1_1.weight)
        nn.init.kaiming_normal_(self.conv1X1_2.weight)

    def forward(self, x):
        x = self.conv1x1_1(x)
        x = self.relu6(x)
        x = self.dsconv(x)
        x = self.relu6(x)
        x = self.conv1X1_2(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = BottleneckResidualBlock(3, 64, 2, 6).cuda()
    summary(model, (3, 224, 224))