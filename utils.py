import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fftpack import dct


def dct_2d(x):
    y = dct(dct(x, axis=1), axis=2)
    return y


def idct_2d(y):
    y = dct(dct(y, axis=1), axis=2)
    return y


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
