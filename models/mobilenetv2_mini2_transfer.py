import torch
import torch.nn as nn

from models.mobilenetv2_mini2 import MobilenetV2Mini


class MobilenetV2MiniTransfer(MobilenetV2Mini):
    def __init__(self, num_age_classes, num_gender_classes, state_dict_path=None):
        super(MobilenetV2MiniTransfer, self).__init__(num_age_classes=num_age_classes, num_gender_classes=num_gender_classes)
        self.model = MobilenetV2Mini(num_age_classes, num_gender_classes)
        if state_dict_path is not None:
            self.model.load_state_dict(torch.load(state_dict_path))

        self.model.classifier_gender = torch.nn.Sequential(
            nn.Linear(640, 640),
            nn.LeakyReLU(inplace=True),
            nn.Linear(640, 640),
            nn.LeakyReLU(inplace=True),
            nn.Linear(640, self.num_gender_classes)
        )

        self._initialize_classifier_gender_weight()

    def _initialize_classifier_gender_weight(self):
        for m in self.model.classifier_gender.modules():
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
        age, gender = self.model(x)

        return age, gender
