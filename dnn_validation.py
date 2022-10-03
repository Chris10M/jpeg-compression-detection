from genericpath import isfile
import cv2
import torch
import os
import csv
import mediafire_dl

from dataset import JPEGDatasetTest
from upsampler import Upsamper
from train import JPEGCompressionModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    if not os.path.isfile('saved_models/checkpoint.ckpt'):
        url = 'https://www.mediafire.com/file/ec224d18v0ufgj0/sample-mnist-epoch%253D08-val_loss%253D4.83.ckpt'
        os.makedirs('saved_models', exist_ok=True)
        mediafire_dl.download(url, 'saved_models/checkpoint.ckpt', quiet=False)

    model = JPEGCompressionModel.load_from_checkpoint("saved_models/checkpoint.ckpt")
    model.eval()
    model = model.to(device)

    val_ds = JPEGDatasetTest('data/val_dataset', five_crop=True)
    
    upsampler = Upsamper()

    rows = list()
    for image_path, x in val_ds:
        x = x.to(device)

        recon_x = upsampler(x)
        x = (recon_x.float() - x) / 255

        x = x.permute(0, 3, 1, 2)

        with torch.no_grad():
            c, r = model(x)
        
        y_hat = r.mean(-1).cpu().item()
        QF = min(int(round(y_hat, 2)), 100)
        jpeg_compressed = c.mean().item() > 0.5

        rows.append([image_path, jpeg_compressed, QF])

        image = cv2.imread(image_path)

        print(f'image_path{image_path} jpeg_compressed: {jpeg_compressed} QF: {QF}')
        cv2.imshow('image', image)    
        cv2.waitKey(20)


    HEADERS = [['image_path', 'jpeg_compressed', 'QF']]
    rows = HEADERS + rows

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/dnn.csv', 'w') as csv_file:
        csv.writer(csv_file).writerows(rows)




if __name__ == '__main__':
    main()
