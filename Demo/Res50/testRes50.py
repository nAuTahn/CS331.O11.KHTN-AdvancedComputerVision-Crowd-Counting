from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import pandas as pd

from Res50.models.CC import CrowdCounter
from Res50.config import cfg
from Res50.misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],
            [0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
pil_to_tensor = standard_transforms.ToTensor()

model_path = 'Res50/all_ep_55_mae_7.9_mse_12.5.pth'

def test(img):
    net = CrowdCounter([0], 'Res50')
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    if img.mode == 'L':
        img = img.convert('RGB')

    img = img_transform(img)

    pred_map = []
    with torch.no_grad():
        img = Variable(img[None, :, :, :]).cuda()
        pred_map = net.test_forward(img)
 
    pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
    pred = np.sum(pred_map)/100.0    

    return pred
        
