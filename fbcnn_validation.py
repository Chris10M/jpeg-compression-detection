import os
if not os.path.isdir('FBCNN'):
    os.system('git clone https://github.com/jiaxi-jiang/FBCNN.git')

import cv2
import csv
import torch
import requests
import numpy as np
import os.path
import torchvision.transforms as transforms
from PIL import Image

from FBCNN.models.network_fbcnn import FBCNN as net
from FBCNN.utils import utils_image as util
from dataset import JPEGDatasetTest


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name):
    model_path = os.path.join('FBCNN', 'model_zoo', model_name)
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{}'.format(os.path.basename(model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {model_path}')
        open(model_path, 'wb').write(r.content)    

    return model_path

class Model:
    def __init__(self) -> None:
        n_channels = 3
        nc = [64,128,256,512]
        nb = 4

        model_name = 'fbcnn_color.pth'
        model_path = load_model(model_name)

        model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        self.model = model.to(device)

        self.n_channels = n_channels
        self.random_crop = transforms.RandomCrop((256, 256))

    def __call__(self, image_path):        
        img = Image.open(image_path).convert('RGB')
        img = self.random_crop(img)
        img_L_np = np.array(img)
        
        img_L = util.uint2tensor4(img_L_np)
        img_L = img_L.to(device)

        with torch.no_grad():
            _, QF = self.model(img_L)
            QF = 1 - QF
            QF = round(float(QF * 100))

        return QF


def main():
    model = Model()
    val_ds = JPEGDatasetTest('data/val_dataset', five_crop=False)
    
    rows = list()
    for image_path, x in val_ds:
        QF = model(image_path)
        image = cv2.imread(image_path)

        rows.append([image_path, '', QF])

        print(f'image_path: {image_path} QF: {QF}')
        cv2.imshow('image', image)    
        cv2.waitKey(20)


    HEADERS = [['image_path', 'jpeg_compressed', 'QF']]
    rows = HEADERS + rows

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/fbcnn.csv', 'w') as csv_file:
        csv.writer(csv_file).writerows(rows)



if __name__ == '__main__':
    main()
