#!/usr/bin/env python

import numpy as np
#import pytiff
from tifffile import imread as tif_imread

# import itertools
import argparse
import logging

# Need this with RTX 3000 series GPU , TF 2.4.0 installed via pip and CUDA 11, cudnn 8 via conda
# otherwise it will say no algorithm worked. very bad.
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)

import cv2
import os

from skimage.filters import median, unsharp_mask
from skimage.morphology import dilation, disk

from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D

import glob
#import logging
#logger = logging.getLogger("stardist")

def get_input_output(args):
  """ Generate input and output paths from command line args """
  if args.input is not None:
    input_path = args.input
  else:
    full_region = f'{args.sample_id}_reg{args.region_num}'
    srch_path = f'{args.data_home}/{full_region}/images/*DAPI*.tif'
    print(f'Searching {srch_path}')
    dapi_images = sorted(glob.glob(srch_path))
    input_path = dapi_images[0]
    print(f'Got input path: {input_path}')

  assert os.path.exists(input_path), f"Input {input_path} does not exist"

  if args.output is not None:
    label_out = args.output
  else:
    full_region = f'{args.sample_id}_reg{args.region_num}'
    label_out = f'{args.data_home}/{full_region}/{full_region}_2_stardist.tif' 

  return input_path, label_out

def main(args):

  image_path, label_out = get_input_output(args)

  # for converting int32 to 4-channel uint8: https://stackoverflow.com/a/25298780 
  dt=np.dtype((np.int32, {'f0':(np.uint8,0),'f1':(np.uint8,1),'f2':(np.uint8,2), 'f3':(np.uint8,3)}))

  #
  # if not os.path.isdir(args.output):
  #   os.makedirs(args.output)

  size = args.size

  logger.info(f'opening file {image_path}')

  if args.HandE:
    logger.info('using opencv')
    img = cv2.imread(image_path, -1)[:,:,::-1] # CV2 uses BGR channel order; reverse it to RGB.

  else:
    logger.info('using tifffile')
    img = tif_imread(image_path)
    # with pytiff.Tiff(image_path, 'r') as handle:
    #   page = handle.pages[args.page] 
    #   img = page[:]


  logger.info(f'loaded image shape: {img.shape} type: {img.dtype} min/max: {img.min()} {img.max()}')

  if args.HandE:
    logger.info('using StarDist pretrained H&E model')
    model = StarDist2D.from_pretrained('2D_versatile_he')
  else:
    logger.info(f'using local StarDist model: {args.model_basedir}/{args.model_name}')
    model = StarDist2D(None, name=args.model_name, basedir=args.model_basedir)

  if args.HandE:
    logger.info('H&E input: not quantile normalizing')
  else:
    value_max = np.quantile(img, args.quantile_max)
    logger.info(f'Normalizing image to {args.quantile_max}-quantile: {value_max}')

    img[img > value_max] = value_max
    img = img.astype(np.float32) / value_max

  if args.factor is not None:
    #target_size = (int(img.shape[0] * args.factor), int(img.shape[0] * args.factor))
    img = cv2.resize(img, dsize=(0,0), fx=args.factor, fy=args.factor, interpolation=cv2.INTER_CUBIC)
    logger.info(f'scaled image: {img.shape}')

  h, w = img.shape[:2]
  n_y = int(np.ceil(h / size))
  n_x = int(np.ceil(w / size))
  logger.info(f'cutting {h}, {w} into {n_y} {n_x}')


  if args.unsharp_mask and not args.HandE:
    r = args.unsharp_mask_radius
    amount = args.unsharp_mask_amount
    logger.info(f'Applying unsharp mask filter r={r} amount={amount}')
    img = unsharp_mask(img, radius=r, amount=amount)
  if args.unsharp_mask and args.HandE:
    logger.info('Unsharp mask was set but the image is H&E')

  if args.HandE:
    logger.info('H&E input: skip median filter')
  else:
    logger.info('Applying median filter')
    img = median(img, selem=disk(args.median_filter_selem_size))


  if args.debugging:
    db_out = f'{args.output}/{os.path.basename(args.input)}_debugging.tif'
    logger.info(f'Debugging: saving processed input image to {db_out} before csbdeep normalization')
    logger.info(f'Debugging: image stats: {img.shape} {img.min()} {img.max()} {img.dtype}')
    cv2.imwrite(db_out, (img*255).astype(np.uint8))


  logger.info('Applying csbdeep normalize')
  img = normalize(img, axis=(0,1))
  logger.info(f'Debugging: post normalization image stats: {img.shape} {img.min()} {img.max()} {img.dtype}')
  logger.info(f'preprocessed image shape: {img.shape} type: {img.dtype} min/max: {img.min()} {img.max()} ')

  logger.info('working....')

  axes = 'YXC' if args.HandE else 'YX'
  n_tiles = (4,4,1) if args.HandE else (4,4)
  labels, _ = model.predict_instances_big(img, axes=axes, block_size=512, min_overlap=196, n_tiles=n_tiles)
  logger.info(f'StarDist2D returned: {labels.shape}, {labels.max()} instances {labels.dtype}')

  logger.info(f'Converting int32 to 3-channel uint8')
  labels = labels.view(dtype=dt)
  # Assume we have no more than 256 ** 3 (=16,777,216) nuclei , so we only take the first 24 bits
  labels = np.dstack([ labels['f0'], labels['f1'], labels['f2'] ])
  logger.info(f'Converted image: {labels.shape} {labels.dtype}')

  logger.info(f'Writing to {label_out}: ')
  cv2.imwrite(label_out, labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_home',  type=str, help='The root data directory for a project')
  parser.add_argument('--sample_id',  type=str, help='Name of the sample to process')
  parser.add_argument('--region_num', type=int, help='The region number to process. If the sample was scanned with 1 region, pass 0')

  parser.add_argument('-i', dest='input', type=str, default=None, 
    help='A tiff image. Byasses the project/sample_id/region_id path building process')

  parser.add_argument('-o', dest='output', type=str, default=None,
    help='A directory. Byasses the project/sample_id/region_id path building process')

  parser.add_argument('--page', default=0, type=int,
                      help = '0-indexed page to use for the DAPI channel data.')

  # parser.add_argument('--subtract_page', default=None, type=int,
  #                     help = '0-indexed page to subtract from the DAPI channel data.')

  parser.add_argument('--quantile_max', default=0.99, type=float,
                      help = 'The percentile value to use for max normalization. Values above are \
                              squished.')

  parser.add_argument('--size', default=256, type=int,
                      help = 'Pixel dimensions for processing through StarDist2D. \
                              n_tiles are inferred from this value and the input size.')

  # Note on factor: the CODEX microscope scans at a weird resolution ~0.33 um/px or something like that.
  # Depending on the tissue and average nucleus size this may/may not affect the performance
  # We basically want to roughly match the average size (in pixels) of nuclei in the processed image to those that the 
  # model was trained with.
  parser.add_argument('--factor', default=None, type=float,
                      help = 'The up (factor > 1) or down (0 < factor < 1) factor to use for resizing prior to analysis.')

  parser.add_argument('--model_name', default='2D_dsb2018_codex2')
  parser.add_argument('--model_basedir', default='stardist_models')
  parser.add_argument('--HandE', action='store_true')

  parser.add_argument('--unsharp_mask', action='store_true',
                      help = 'whether to apply an unsharp mask filter '+\
                             '(https://scikit-image.org/docs/dev/auto_examples/filters/plot_unsharp_mask.html)')
  parser.add_argument('--unsharp_mask_radius', default=3, type=float,
                      help = 'radius parameter for unsharp mask filter')
  parser.add_argument('--unsharp_mask_amount', default=1, type=float,
                      help = 'amount parameter for unsharp mask filter')

  parser.add_argument('--median_filter_selem_size', default=5, type=int,
                      help = 'disk structuring element size for median filtering')

  parser.add_argument('--debugging', action='store_true', 
                      help = 'Whether to calculate and save some intermediates / alternative pipeline outputs.')

  args = parser.parse_args()

  logger = logging.getLogger("CODEX")
  logger.setLevel("INFO")
  ch = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  logger.info('Starting')
  for k, v in args.__dict__.items():
    logger.info(f'{k}: {v}')

  main(args)
