import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

from upsampler import Upsamper
from train import JPEGCompressionModel


upsampler = Upsamper()


def preprocess_image(image_path):
    random_crop = transforms.RandomCrop((224, 224))

    img = Image.open(image_path)
    img = random_crop(img)

    x = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    QF = np.random.randint(5, 95)
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, QF]
    result, encimg = cv2.imencode('.jpg', x, encode_param)
    x = cv2.imdecode(encimg, 1)

    cv2.imshow('x', x)

    x = torch.tensor(x)[None, ...]
    recon_x = upsampler(x)

    cv2.imshow('recon_x', recon_x[0].cpu().numpy())

    x = (recon_x.float() - x) / 255

    cv2.imshow('diff', x[0].cpu().numpy() * 2)

    x = x.permute(0, 3, 1, 2)

    # img = torch.tensor(img, dtype=torch.float32)
    # x = torch.tensor(x, dtype=torch.float32)

    return x, QF


def main():
    model = JPEGCompressionModel.load_from_checkpoint("saved_models/sample-mnist-epoch=05-val_loss=6.55.ckpt")
    model.eval()
    
    while True:
        x, qf = preprocess_image('data/DIV2K_valid_HR/0802.png')
        
        with torch.no_grad():
            c, r = model(x)

        print(c, r, qf)
        c = cv2.waitKey(0)

        if ord('q') == c:
            break






if __name__ == '__main__':
    main()
