import torch

from torchvision.models import mobilenet_v2


class CustomMobilenetV2(torch.nn.Module):
    def __init__(self, num_classes, freeze_convs=False):
        super(CustomMobilenetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=True)
        if freeze_convs:
            self._freeze_pretrained_conv_params()
        self.model.classifier[-1] = torch.nn.Linear(1280, num_classes)

    def _freeze_pretrained_conv_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    from torchsummary import summary
    model = CustomMobilenetV2(26, True).cuda()
    summary(model, (3, 224, 224))