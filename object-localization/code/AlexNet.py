import torch.nn as nn
import torchvision.models as models
import torch

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# model = torch.hub.load_state_dict_from_url(model_urls["alexnet"])
# # print(model.keys())
# for k,v in model.items():
#     if 'features' in k:
#         print(k)

class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        # TODO (Q1.1): Define model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True)
        )


        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )


    def forward(self, x):
        # TODO (Q1.1): Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x

class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model
        self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
        nn.ReLU(inplace= True),
        nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=False),
        nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        nn.ReLU(inplace= True),
        nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=False),
        nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace= True),
        nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace= True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace = True)
        )


        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )


    def forward(self, x):
        # TODO (Q1.7): Define forward pass
        x = self.features(x)
        x = self.classifier(x)

        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    base_state_dict = torch.hub.load_state_dict_from_url(model_urls["alexnet"])
    model_state_dict = model.state_dict()
    # print(model_state_dict.parameters)
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not
    if pretrained:
        for k, v in base_state_dict.items():
            if 'features' in k:
                model_state_dict[k] = v
        for k, v in model_state_dict.items():
            if 'classifier' in k and 'weight' in k:
                model_state_dict[k] = nn.init.xavier_uniform_(v)

    else:
        for k, v in model_state_dict.items():
            if 'weight' in k:
                model_state_dict[k] = nn.init.xavier_uniform_(v)

    model.load_state_dict(model_state_dict)
    return model


# model = localizer_alexnet(pretrained = True)


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not
    base_state_dict = torch.hub.load_state_dict_from_url(model_urls["alexnet"])
    model_state_dict = model.state_dict()
    # print(model_state_dict.parameters)
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not
    if pretrained:
        for k, v in base_state_dict.items():
            if 'features' in k:
                model_state_dict[k] = v
        for k, v in model_state_dict.items():
            if 'classifier' in k and 'weight' in k:
                model_state_dict[k] = nn.init.xavier_uniform_(v)

    else:
        for k, v in model_state_dict.items():
            if 'weight' in k:
                model_state_dict[k] = nn.init.xavier_uniform_(v)


    model.load_state_dict(model_state_dict)
    return model

