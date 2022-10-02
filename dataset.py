import cv2
import numpy as np
import torch
import torch_dct as dct
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import split_img_into_blocks, dct_2d, idct_2d


def preprocess_image(bgr_image):
    h, w = bgr_image.shape[:2]
    bgr_image = bgr_image[:h//8 * 8, :w//8 * 8]
    
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycrcb)

    image_chunks = split_img_into_blocks(y)
    y = torch.tensor(image_chunks.astype(np.float32) / 255)
    y = dct.dct_2d(y)[..., None]

    image_chunks = split_img_into_blocks(cr)
    cr = torch.tensor(image_chunks.astype(np.float32) / 255)
    cr = dct.dct_2d(cr)[..., None]

    image_chunks = split_img_into_blocks(cb)
    cb = torch.tensor(image_chunks.astype(np.float32) / 255)
    cb = dct.dct_2d(cb)[..., None]
    
    X = torch.cat([y, cb, cr], -1)

    return X


class JPEGDatasetTrain(Dataset):
    def __init__(self, root, mode) -> None:
        self.root = root

        assert mode in ['train', 'val']
        self.file_paths = list()

        rep_factor = 1
        if 'train' in mode:
            rep_factor = 100

        for _ in range(rep_factor):
            for root, _, filenames in os.walk(self.root):
                self.file_paths.extend([os.path.join(root, filename) for filename in filenames])

        self.len = len(self.file_paths)
        self.random_crop = transforms.RandomCrop((224, 224))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.file_paths[index]

        img = Image.open(image_path)
        img = self.random_crop(img)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        QF = np.random.randint(5, 95)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, QF]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        x = cv2.imdecode(encimg, 1)

        # img = torch.tensor(img, dtype=torch.float32)
        # x = torch.tensor(x, dtype=torch.float32)

        return x, QF


class JPEGDatasetTest(Dataset):
    def __init__(self, root, five_crop=False) -> None:
        self.root = root
        self.file_paths = list()

        for root, _, filenames in os.walk(self.root):
            self.file_paths.extend([os.path.join(root, filename) for filename in filenames])

        self.len = len(self.file_paths)

        self.five_crop = five_crop        
        if five_crop:
            self.random_crop = transforms.FiveCrop((224, 224))
        else:
            self.random_crop = transforms.RandomCrop((224, 224))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.file_paths[index]

        img = Image.open(image_path)
        img = self.random_crop(img)
                
        if not self.five_crop:
            x = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            x = torch.tensor(x)[None, ...]
        else:
            xs = list()
            for im in img:
                x = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                x = torch.tensor(x)[None, ...]
                xs.append(x)

            x = torch.cat(xs)

        return image_path, x


def main():
    image = cv2.imread('test.png')
    x = preprocess_image(image)
    # preprocess_image()

    print(x.shape)
    
    print(image_chunks.shape)

    
    X = dct.dct_2d(x)

    

    y = dct.idct_2d(X) 
    
    # truncate values in the frequency domain
    frequency = cv2.dct(image_chunk)
    frequency = q * np.round(frequency / q)

    # restore to the values domain and clip by image domain
    restored = cv2.idct(frequency)
    restored = np.clip(restored, 0, 255).astype(np.uint8)


    print((np.abs(x - y)).sum())

    assert (np.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance


if __name__ == '__main__':
    main()
