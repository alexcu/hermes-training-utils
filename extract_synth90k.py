#!/usr/bin/env python

# Extracts Synth90k ground truths from MATLAB binary
# Usage: python extract_synth90k.py /path/to/gt.mat

import numpy as np
import pandas as pd
import scipy.io
import sys
import os

filepath = sys.argv[1]
mat = scipy.io.loadmat(filepath)
mat = {k:v for k, v in mat.items() if k[0] != '_'}
DF = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})

def extract_image(index):
  img_path = DF["imnames"][index][0]
  bbox = DF["wordBB"][index]
  words = DF["txt"][index]
  annotations = []
  bbox_x1s = bbox[0][0]
  bbox_x2s = bbox[0][1]
  bbox_x3s = bbox[0][2]
  bbox_x4s = bbox[0][3]
  bbox_y1s = bbox[1][0]
  bbox_y2s = bbox[1][1]
  bbox_y3s = bbox[1][2]
  bbox_y4s = bbox[1][3]
  if isinstance(bbox_x1s, np.float32):
    bbox_x1s = [bbox[0][0]]
    bbox_x2s = [bbox[0][1]]
    bbox_x3s = [bbox[0][2]]
    bbox_x4s = [bbox[0][3]]
    bbox_y1s = [bbox[1][0]]
    bbox_y2s = [bbox[1][1]]
    bbox_y3s = [bbox[1][2]]
    bbox_y4s = [bbox[1][3]]
  assert len(bbox_x1s) == \
         len(bbox_x2s) == \
         len(bbox_x3s) == \
         len(bbox_x4s) == \
         len(bbox_y1s) == \
         len(bbox_y2s) == \
         len(bbox_y3s) == \
         len(bbox_y4s), "Dimensions must be the same!"
  for i in range(len(bbox_x1s)):
    poly = np.array([
      [bbox_x1s[i], bbox_y1s[i]],
      [bbox_x2s[i], bbox_y2s[i]],
      [bbox_x3s[i], bbox_y3s[i]],
      [bbox_x4s[i], bbox_y4s[i]]
    ])
    bbox = [
      poly.min(axis = 0),
      poly.max(axis = 0)
    ]
    annotations.append(bbox)
  return (img_path, annotations)

for idx in range(len(DF)):
  dirname = os.path.dirname(filepath)
  img_path, bboxes = extract_image(idx)
  img_path = os.path.join(dirname, img_path)
  for box in bboxes:
    #img_path,x1,y1,x2,y2
    x1 = int(box[0][0])
    y1 = int(box[0][1])
    x2 = int(box[1][0])
    y2 = int(box[1][1])
    print("%s,%i,%i,%i,%i,text" % (img_path, x1, y1, x2, y2))
