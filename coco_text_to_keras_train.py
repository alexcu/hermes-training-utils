#!/usr/bin/env python

import json
import sys
import random
import os
from shutil import copyfile

assert len(sys.argv) - 1 >= 3, "Two arguments required: path to coco images, path to coco JSON, path to output. Optional fourth argument is the number of annotions to use"
json_file = sys.argv[1]
imgs_dir = sys.argv[2]
out_dir = sys.argv[3]
numb_of_anns = None
if len(sys.argv) - 1 > 3:
  numb_of_anns = int(sys.argv[4])

print "Loading JSON..."
with open(json_file) as json_file:
  json_data = json.load(json_file)

# Split images between training and evaluation
def get_set(name):
    set = [img for img in json_data['imgs'].values() if img['set'] == name]
    print "There are %i %s images" % (len(set), name)
    return set

def annotations_for_set(set, class_name):
  set_name = set[0]['set']
  annotations = []
  for img_id, ann_ids in json_data['imgToAnns'].iteritems():
    for ann_id in ann_ids:
      annotation = json_data['anns'][str(ann_id)]
      filename = json_data['imgs'][str(img_id)]['file_name']
      annotation['filename'] = filename
      annotations.append(annotation)
  print "There are %i '%s' annotations in the %s set" % (len(annotations),
                                                         class_name,
                                                         set_name)
  return annotations


train_imgs, test_imgs, val_imgs = (get_set('train'),
                                   get_set('test'),
                                   get_set('val'))

train_anns, test_anns, val_anns = (annotations_for_set(train_imgs, 'machine printed'),
                                   annotations_for_set(test_imgs, 'machine printed'),
                                   annotations_for_set(val_imgs, 'machine printed'))

lookups = ((train_anns, 'train'),
           (test_anns, 'test'),
           (val_anns, 'val'))

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

for anns, type in lookups:
  # Condense at random
  if numb_of_anns is not None:
    print "Randomly selecting %d of them for %s" % (numb_of_anns, type)
    random.shuffle(anns)
    anns = anns[:numb_of_anns]
  # Write out
  img_out_dir = os.path.join(out_dir, type)
  txt_out = os.path.join(out_dir, "%s.txt" % type)
  os.makedirs(img_out_dir)
  # Copy the image over if it does not exist
  for ann in anns:
    src_img = os.path.join(imgs_dir, ann['filename'])
    dst_img = os.path.join(img_out_dir, ann['filename'])
    if not os.path.exists(dst_img):
      print "Copying %s to %s" % (src_img, dst_img)
      copyfile(src_img, dst_img)
    # Write record to file
    with open(txt_out, "a") as text_file:
      line = ','.join([
        src_img,
        str(ann['bbox'][0]), # x
        str(ann['bbox'][1]), # y
        str(ann['bbox'][2]), # w
        str(ann['bbox'][3]), # h
        'text'
      ])
      text_file.write("%s\n" % line)
