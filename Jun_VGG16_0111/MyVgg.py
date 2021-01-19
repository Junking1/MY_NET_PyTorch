
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

class  MyVGG(nn.Module):
    def __init__(self, features):
        super(MyVGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def make_layer(cfg):
    layer = []
    in_channels = 3
    for i in cfg:
        if i == 'M':
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2 = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
            layer += [conv2, nn.ReLU(inplace=True)]
            in_channels = i
    return nn.Sequential(*layer)

cfgs = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
              512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
              512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG16(**kwargs):

    Vgg16Net = MyVGG((make_layer(cfgs['VGG16'])), **kwargs)
    # state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
    #                                       model_dir="./model_data")
    # Vgg16Net.load_state_dict(state_dict)
    return Vgg16Net

