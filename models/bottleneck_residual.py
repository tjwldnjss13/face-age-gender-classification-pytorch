import torch.nn as nn

from models.dsconv import DConv


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(BottleneckResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_factor = expansion_factor

        self.mid_channels = in_channels * expansion_factor
        self.conv1x1_1 = nn.Conv2d(in_channels, self.mid_channels, 1, 1, 0)
        self.dsconv = DConv(self.mid_channels, 3, stride, 1)
        self.conv1X1_2 = nn.Conv2d(self.mid_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.relu6 = nn.ReLU6(True)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d()
        if stride == 1:
            self.conv_residual = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.conv_residual = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

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
        # nn.init.kaiming_normal_(self.conv1x1_1.weight)
        # nn.init.kaiming_normal_(self.conv1X1_2.weight)
        # nn.init.kaiming_normal_(self.conv_residual.weight)

    def forward(self, x):
        residual = x
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.dsconv(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.conv1X1_2(x)
        x = self.bn3(x)

        residual = self.conv_residual(residual)
        x += residual

        x = self.lrelu(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = BottleneckResidualBlock(3, 64, 2, 6).cuda()
    summary(model, (3, 224, 224))