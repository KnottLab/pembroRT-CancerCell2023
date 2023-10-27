#!/usr/bin/env python
"""
We get segmentations from stardist as a label image

Process the unique cells into a binary mask, then apply a dilation to get the membranes
"""

import cv2
#import pytiff
from tifffile import imread as tif_imread
import zarr
import numpy as np
import pandas as pd

import argparse
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction
from skimage.measure import regionprops

import os
import glob
import sys

# def tissue_mask(dapi_file, scale=0.5):
#   with pytiff.Tiff(dapi_file, 'r') as f:
#     tissue = f.pages[0][:]

#   tissue = cv2.resize(tissue, dsize=(0,0), fx=scale, fy=scale)
#   thr = threshold_otsu(tissue.ravel())
#   tissue = tissue > thr

#   tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement('disk', (40,40)))
#   seed = np.ones_like(tissue)
#   tissue[ : ,0] = 0
#   tissue[ : ,-1] = 0
#   tissue[ 0 ,:] = 0
#   tissue[ -1 ,:] = 0
#   seed[ : ,0] = 0
#   seed[ : ,-1] = 0
#   seed[ 0 ,:] = 0
#   seed[ -1 ,:] = 0
#   tissue = reconstruction(seed, tissue, method='erosion')
#   tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement('disk', (20,20)))

def tissue_mask(dapi_file, scale):
  # with pytiff.Tiff(dapi_file, 'r') as f:
  #   dapi = f.pages[0][:]
  dapi = tif_imread(dapi_file)
  return np.ones_like(dapi, dtype=np.bool)


def rgb2label(rgb, tissue):
  h,w = rgb.shape[:2]
  img_exp = np.concatenate([rgb, np.zeros((h,w,1),dtype=np.uint8)], axis=-1)
  s = img_exp.view(dtype=np.uint32)[:,:,0].copy().astype(np.float32)

  s[tissue==0] = 0

  d = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
  di = d - s
  s[di > 0] = 0
  return s


def nuclei_props(s):
  props = regionprops(s.astype(np.uint32))
  cell_ID = []
  X = []
  Y = []
  Size = []
  for i, p in enumerate(props):
    cell_ID.append(f'cell_{i}')
    X.append(int(p.centroid[1]))
    Y.append(int(p.centroid[0]))
    Size.append(p.area)
  
  df = pd.DataFrame({'X': X, 'Y': Y, 'Size': Size}, index=cell_ID)
  df.index.name = 'cell_ID'
  return df


def membrane_mask(s):
  kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
  m = s.copy()
  for k in range(5):
      mt = cv2.dilate(m, kern)
      mt[m!=0] = m[m!=0] #reset places where overlaps happened
      m = mt
      
  mt = cv2.dilate(m,kern);
  m[(mt-m)>0] = 0;
  return m

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file')
  parser.add_argument('--tissue', default=None, type=str)
  parser.add_argument('--clobber', action='store_true')
  # parser.add_argument('dapi_file')

  ARGS = parser.parse_args()

  # generate some opinionated output file names
  if ARGS.tissue is None:
    tissue_out = ARGS.input_file.replace('_2_stardist', '_1_tissue')
  else:
    tissue_out = f'using existing tissue image: {ARGS.tissue}'
  nuclei_out = ARGS.input_file.replace('_2_stardist', '_2_nuclei')
  membrane_out = ARGS.input_file.replace('_2_stardist', '_2_membrane')
  centroid_out = ARGS.input_file.replace('_2_stardist.tif', '_2_centroids.csv')

  if os.path.exists(centroid_out) and not ARGS.clobber:
    print(f'Centroid output exists: {centroid_out}. Exiting.')
    sys.exit(0)

  print(f'tissue --> {tissue_out}')
  print(f'nuclei --> {nuclei_out}')
  print(f'membrane --> {membrane_out}')
  print(f'centroid table --> {centroid_out}')

  # infer a DAPI image, we need it to match the sizes
  sample_dir = os.path.split(ARGS.input_file)[0]  
  srch = f'{sample_dir}/images/*DAPI*' 
  print(f'searching for DAPI at: {srch}')
  candidates = sorted(glob.glob(srch))

  if len(candidates) == 0:
    print(f'Found no DAPI images under {srch}')
    sys.exit(1)
  
  dapi_file = candidates[0]
  print(f'using DAPI file: {dapi_file}')

  store = tif_imread(dapi_file, aszarr=True)
  z = zarr.open(store, mode='r')
  ref_rows, ref_cols = z.shape[:2]
  store.close()
  print(f'REFERENCE TISSUE image sized: {ref_rows, ref_cols}')


  if (ARGS.tissue is not None) and os.path.exists(ARGS.tissue):
    print(f'Loading tissue mask from: {ARGS.tissue}')
    tissue = tif_imread(ARGS.tissue)
    print(f'resizing tissue image from {tissue.shape}')
    tissue = cv2.resize(tissue, dsize=(ref_cols, ref_rows), interpolation=cv2.INTER_NEAREST)
  else:
    tissue = np.ones((ref_rows, ref_cols), dtype=np.uint8)*255

  image = tif_imread(ARGS.input_file) # the stardist segmentation encodings
  print(f'NUCLEI image sized: {image.shape}')

  print('resizing')
  image = cv2.resize(image, dsize=(ref_cols,ref_rows), interpolation=cv2.INTER_NEAREST)
  print(f'new size: {image.shape}')

  if (ARGS.tissue is None) and not os.path.exists(tissue_out):
    #tissue = np.ones_like(tissue, dtype=np.uint8)*255
    cv2.imwrite(tissue_out, tissue)

  print(f'Creating labels from the nuclei')
  labels = rgb2label(image, tissue)
  label_mask = labels>0

  print(f'Writing nuclei image')
  cv2.imwrite(nuclei_out, label_mask.astype(np.uint8)*255)

  print(f'Calculating nuclei shape props')
  prop_df = nuclei_props(labels)
  print(f'centroids: {prop_df.shape[0]}')
  prop_df.to_csv(centroid_out)

  print(f'creating membrane mask')
  membrane = membrane_mask(labels)
  membrane_mask = membrane > 0
  cv2.imwrite(membrane_out, membrane_mask.astype(np.uint8)*255)

