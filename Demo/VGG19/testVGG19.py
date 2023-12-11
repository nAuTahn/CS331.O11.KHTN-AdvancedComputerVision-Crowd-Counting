import torch
import os
import numpy as np
from VGG19.datasets.crowd import Crowd
from VGG19.models.vgg import vgg19
import torchvision.transforms.functional as F
from torchvision import transforms

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def test(img):
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    ckpt = torch.load('VGG19/VGG19-SHHB.pth', device)

    model.load_state_dict(ckpt)

    img = trans(img)
    inputs = img.to(device)

    enumber = 0
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        enumber = torch.sum(outputs).item()

    return enumber


