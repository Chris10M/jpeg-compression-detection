import os
import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer


class Upsamper:
    def __init__(self) -> None:
        model_name = 'RealESRGAN_x2plus'

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        
        os.makedirs('weights', exist_ok=True)

        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True)

    def __call__(self, img_tensor):
        outputs = list()
        for img in img_tensor.cpu().numpy():
            output, _ = self.upsampler.enhance(img, outscale=1)
            outputs.append(output)

        outputs = torch.from_numpy(np.array(outputs)).to(img_tensor.device)

        return outputs


def main():
    upsampler = Upsamper()
    x = np.random.rand(256, 256, 3) * 255
    x = x.astype(dtype=np.uint8)

    y = upsampler(x)


if __name__ == '__main__':
    main()