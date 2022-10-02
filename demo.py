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

    cv2.imshow('diff', x[0].cpu().numpy())

    x = x.permute(0, 3, 1, 2)

    # img = torch.tensor(img, dtype=torch.float32)
    # x = torch.tensor(x, dtype=torch.float32)

    return x, QF


def main():
    model = JPEGCompressionModel.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=2-step=939.ckpt")
    model.eval()
    
    x, qf = preprocess_image('/media/ssd/christen-rnd/Experiments/jpeg-compression-detection/data/DIV2K_valid_HR/0802.png')
    
    with torch.no_grad():
        y_hat = model(x)

    print(y_hat, qf)
    cv2.waitKey(0)






if __name__ == '__main__':
    main()
