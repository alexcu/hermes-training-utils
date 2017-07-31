#!/usr/bin/env python

import os
import sys
import random
from glob import glob
from shutil import copyfile

def csv_to_record(csvrow, dst):
    # Record format for keras-frcnn
    data = csvrow.split(",")
    class_name = data[0]
    x1 = data[1]
    y1 = data[2]
    x2 = data[3]
    y2 = data[4]
    return ",".join([dst, x1, y1, x2, y2, class_name])

def csv_to_record_lines(src, dst):
    return "\n".join([csv_to_bib(row, dst) for row in list(open(src, 'r'))])

def copy_unique_image(filename, out_dir):
    # filename is the original 'XXXX_org.jpg' filename; we want all CSVs
    # and augmented files using this identifier, XXXX
    image_id =  os.path.basename(filename).split("_org.jpg")[0]
    from_dir = os.path.dirname(filename)

    print "Copying image identifier '%s'..." % image_id

    all_related_files = glob("%s/%s*" % (from_dir, image_id))

    copy_text = []

    for file in all_related_files:
        file_basename = os.path.basename(file)
        file_ext = os.path.splitext(file)[1]
        file_dst = "%s/%s" % (out_dir, file_basename)
        print "Copying '%s' to '%s'..." % (file, file_dst)
        if file_ext == "csv":
            copy_text.append(csv_to_record_lines(file, file_dst))
        copyfile(file, file_dst)

    return "\n".join(copy_text)



def process(in_dir, out_dir):
    # Process all files for in directory to out directory
    random.seed()

    all_original_files = glob("%s/*_org.jpg" % in_dir)
    random.shuffle(all_original_files)
    print "There are %i original files:" % len(all_original_files)

    num_train = int(0.9 * len(all_original_files))
    train_files = all_original_files[:num_train]
    val_files = all_original_files[num_train:]

    print "90%% (%i) of these will be used for training" % len(train_files)
    print "10%% (%i) of these will be used for validation" % len(val_files)

    for proc_type, files in [("train", train_files), ("val", val_files)]:
        print "Now processing '%s'..." % proc_type
        out_dir_of_type = "%s/%s" % (out_dir, proc_type)
        if not os.path.exists(out_dir_of_type):
            os.makedirs(out_dir_of_type)
        for file in files:
            outfile = copy_unique_image(file, out_dir_of_type)
            with open("%s/results_%s.txt" % (out_dir, proc_type), "a") as text_file:
                text_file.write(outfile)

if __name__ == "__main__":
    # Must provide in and out dir
    assert len(sys.argv) - 1 >= 2, "Must provide two arguments (in dir and out dir)"

    # Input directory
    in_dir = sys.argv[1]
    assert in_dir != None, "Missing input directory (argv[1])"

    # Setup output
    out_dir = sys.argv[2]
    assert out_dir != None, "Missing output directory (argv[2])"

    process(in_dir, out_dir)
