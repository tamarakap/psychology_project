import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1

vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)


def load_model(num_classes, type='resnet', pretrained = False):
    if type == 'resnet':
        if not pretrained:
            model = InceptionResnetV1(classify=True, num_classes=num_classes).eval()
        else:
            model = InceptionResnetV1(classify=True, pretrained = pretrained, num_classes=num_classes).eval()

    elif type == 'vgg':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes + 1)
    else:
        raise Exception('model could not be loaded')

    return model