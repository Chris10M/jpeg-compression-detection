import cv2
import torch

from dataset import JPEGDatasetTest
from upsampler import Upsamper
from train import JPEGCompressionModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    model = JPEGCompressionModel.load_from_checkpoint("saved_models/sample-mnist-epoch=05-val_loss=6.78.ckpt")
    model.eval()
    model = model.to(device)

    val_ds = JPEGDatasetTest('data/val_dataset', five_crop=True)
    
    upsampler = Upsamper()
    for image_path, x in val_ds:
        x = x.to(device)

        recon_x = upsampler(x)
        x = (recon_x.float() - x) / 255

        x = x.permute(0, 3, 1, 2)

        with torch.no_grad():
            y_hats = model(x)

        y_hat = y_hats.max(-1).values.cpu().item()

        image = cv2.imread(image_path)
        
        print(f'QF: {min(int(round(y_hat, 2)), 100)}')
        cv2.imshow('image', image)    
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
