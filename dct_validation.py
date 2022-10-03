import cv2
import os
import csv
import numpy as np
from dataset import JPEGDatasetTest
from numpy.lib.stride_tricks import as_strided


def split_img_into_blocks(image, block_size=8, flatten=True):
  """Split image into chunks"""
  bsize = image.itemsize
  height, width = image.shape

  num_blocks_height = height // block_size
  num_blocks_width = width // block_size

  assert height % block_size == 0 and width % block_size == 0

  shape = (num_blocks_height, num_blocks_width, block_size, block_size)
  strides = (block_size * width * bsize, block_size * bsize, width * bsize, bsize)

  output = as_strided(image, shape=shape, strides=strides)
  if flatten:
    output = output.reshape(-1, block_size, block_size)
  return output


def is_jpeg_compressed(bgr_image):
    h, w = bgr_image.shape[:2]
    bgr_image = bgr_image[:h//8 * 8, :w//8 * 8]
    
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycrcb)
    image_chunks = split_img_into_blocks(y)

    n_blocks = image_chunks.shape[0]

    count = np.zeros(64)
    for image_chunk in image_chunks:
        frequency = cv2.dct(image_chunk.astype(np.float32))
        frequency = np.abs(np.round(frequency))
        
        a = frequency
        frequency = np.concatenate([np.diagonal(a[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-a.shape[0], a.shape[0])])
        count += frequency
    
    count = count.reshape(-1) 
    count = count / n_blocks

    hf_variance = int(round(np.var(count[48:]), 4) * 1e3)

    if hf_variance:
        return False

    return True


def main():    
    val_ds = JPEGDatasetTest('data/val_dataset', five_crop=False)
    
    rows = list()
    for image_path, x in val_ds:
        image = cv2.imread(image_path)
        
        jpeg_compressed = is_jpeg_compressed(image)
        
        rows.append([image_path, jpeg_compressed, ''])

        print(f'image_path: {image_path} jpeg_compressed: {jpeg_compressed}')
        cv2.imshow('image', image)    
        cv2.waitKey(20)

    HEADERS = [['image_path', 'jpeg_compressed', 'QF']]
    rows = HEADERS + rows

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/dct.csv', 'w') as csv_file:
        csv.writer(csv_file).writerows(rows)


if __name__ == '__main__':
    main()
