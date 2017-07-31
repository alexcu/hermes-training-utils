#!/usr/bin/env python

import tensorflow as tf
import sys
from glob import glob
from scipy import misc
import random
import os

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('input_dir', '', 'Root directory to raw bib dataset.')
FLAGS = flags.FLAGS


def create_tf_example(filename, bibs):
    image = misc.imread(filename)

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()

    # Required sizes
    print "Processing %s" % filename
    height = image.shape[0]
    width = image.shape[1]
    filename = filename
    encoded_image_data = encoded_jpg
    image_format = b'jpeg'

    # Coordinates
    xmins = [bib["x_min"] / width for bib in bibs]
    xmaxs = [bib["x_max"] / width for bib in bibs]
    ymins = [bib["y_min"] / height for bib in bibs]
    ymaxs = [bib["y_max"] / height for bib in bibs]

    classes_text = ['bib'] * len(bibs)
    classes = [1] * len(bibs)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(output_filename,
                     dataset):
    writer = tf.python_io.TFRecordWriter(output_filename)

    for (bibjpg, bibcsv) in dataset:
        bibs = [csv_to_bib(row) for row in list(open(bibcsv, 'r'))]
        if len(bibs) > 0:
            tf_example = create_tf_example(bibjpg, bibs)
            writer.write(tf_example.SerializeToString())

    writer.close()

def csv_to_bib(csvrow):
    data = csvrow.split(",")
    # Ignore 0 as this is always 'bib'
    x_min = float(data[1])
    y_min = float(data[2])
    x_max = float(data[3])
    y_max = float(data[4])
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

def main(_):
    indir = FLAGS.input_dir
    print "Reading in dataset from %s..." % indir
    bibjpg = glob("%s/*.jpg" % indir)
    bibcsv = glob("%s/*.csv" % indir)
    examples_list = zip(bibjpg, bibcsv)

    print "There are %i files in the dataset." % len(bibjpg)

    # Split for training and validation
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    print '%d training and %d validation examples.' % (len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'bib_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'bib_val.record')

    print train_output_path

    create_tf_record(train_output_path, train_examples)
    create_tf_record(val_output_path, val_examples)

if __name__ == '__main__':
    tf.app.run()
