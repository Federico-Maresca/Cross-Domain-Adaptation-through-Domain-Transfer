import os
import argparse
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from ssd.adain.adain_model import Model
import torch.nn.functional as F
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([normalize])

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def adain_transfer(content_batch, style_batch, model_AdaIN):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #work on the batch transformation by using model.generate(...)
    
    n_content = list(content_batch.size())[0] 
    n_style = list(style_batch.size())[0]
    #print("Content batch size:{} \nStyle batch size: {}".format(content_batch.size(),style_batch.size()))
    # process each content image one by one 
    random.seed()
    for i in range(n_content):
        
        j = random.randrange(n_style)

        # batch tensors have shape [32, 3, 300, 300]
        content = content_batch[i, :, :, :]
        style = style_batch[j, :, :, :]

        content = trans(content).unsqueeze(0).to(device)
        style = trans(style).unsqueeze(0).to(device)

        #print("Content size after trans:{}\nStyle size after trans:{} ".format(content.size(),style.size()))
        with torch.no_grad():
              content = F.interpolate(model_AdaIN.generate(content, style, 1), size=[300, 300])
        #print("Content size after denorm:{}".format(content.size()))
        
        content = denorm(content, device)

        # SSD images have been resized to 300 [1, 3, 300, 300]
        # and we copy the 300x300 image matrix to the i'th tensor 
        #in the image tenso batch
        content_batch[i, :, :, :] = content[:, :, :300, :300]  