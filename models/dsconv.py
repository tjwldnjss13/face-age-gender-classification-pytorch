import torch.nn as nn


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=False, batch_norm=False):
        super(DSConv, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.dconv = DConv(in_channels, kernel_size, stride, padding)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        if activation:
            self.lrelu = nn.LeakyReLU(.1, True)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.conv1x1.weight)

    def forward(self, x):
        x = self.dconv(x)
        if self.batch_norm:
            x = self.bn1(x)
        if self.activation:
            x = self.lrelu(x)
        x = self.conv1x1(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.activation:
            x = self.lrelu(x)

        return x


class DConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(DConv, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.dconv.weight)

    def forward(self, x):
        return self.dconv(x)