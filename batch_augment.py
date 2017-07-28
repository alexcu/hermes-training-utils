import os
import sys
import imgaug as ia
import numpy as np
import scipy
import math
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from imgaug import augmenters as iaa
from scipy import misc
from glob import glob
from tempfile import mkstemp


# Resize all our images to SCALE%; map coordinates to new scale
SCALE = 0.2

# How many times to augment one image
AUGMENT_TIMES = 50

def augment(images, image_identifiers, keypoints, out_dir):
    # This is the primary augmentation function

    ###
    # Provide access to images and keypoints
    ###
    def affine():
        # Affine transformation
        TRANSLATE_PCT_RANGE = 0.35
        ROTATION_RANGE = (-45,45)
        SHEAR_RANGE = (-5,5)

        translate_percent = {
            "x": (-TRANSLATE_PCT_RANGE, +TRANSLATE_PCT_RANGE),
            "y": (-TRANSLATE_PCT_RANGE, +TRANSLATE_PCT_RANGE),
        }
        rotate=ROTATION_RANGE
        shear=SHEAR_RANGE
        mode = "edge"

        return iaa.Affine(translate_percent=translate_percent,
                        rotate=rotate,
                        shear=shear,
                        mode=mode)

    def add_neg():
        # Applies a negative to all channels
        return iaa.Add((-45, 0))

    def add_pos():
        # Applies a positive to all channels
        return iaa.Add((0, 45))

    def mul_neg():
        # Multiples all channels by a negative factor
        return iaa.Multiply((0.5, 1))

    def mul_pos():
        # Multiples all channels by a postive factor
        return iaa.Multiply((1, 1.5))

    def blur():
        # Chooses one of three blur methods
        return one_of([
            iaa.GaussianBlur((0, 3.0)),
            iaa.AverageBlur(k=(2, 4)),
            iaa.MedianBlur(k=(3, 5)),
        ])

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    def sometimes(aug, pct = 0.5):
        return iaa.Sometimes(pct, aug)

    def one_of(funcs):
        # Shortcut for iaa.OneOf
        return iaa.OneOf(funcs)

    def keypoints_per_person(kpts):
        # Group one keypoint per person (mod 4)
        return [kpts.keypoints[i:i + 4] for i in range(0, len(kpts.keypoints), 4)]

    def valid_keypoints(kpts, image):
        # Returns any keypoints that are outside the width/height of the image
        width = image.shape[1]
        height = image.shape[0]
        # Group by four (for each person)
        runner_keypoints = keypoints_per_person(kpts)
        # Copy over the "valid" keypoints (assume all are valid)
        valid_keypoints = [e for e in runner_keypoints]
        for kpts in runner_keypoints:
            for k in kpts:
                # If hidden, remove this runner
                if k.x < 0 or k.x > width or k .y < 0 or k.y > height:
                    # Remove from valid if hidden
                    valid_keypoints = [k for k in valid_keypoints if k is not kpts]
                    break
        # Whatever remains becomes the Bib click points for these runners
        return valid_keypoints

    def keypoints_to_rects(kpts):
        # Converts a set of keypoints to rectangles (min/max x/y)
        rects = []
        for kpt in kpts:
            xs = [i.x for i in kpt]
            ys = [i.y for i in kpt]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            rects.append({"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y})
        return rects

    def generate_csv_for_image_kpts(image, image_identifier, kpts):
        # Writes a csv of all rectangles for this image
        image_aug_keypoints = valid_keypoints(kpts, image)
        image_aug_rects = keypoints_to_rects(image_aug_keypoints)
        lines = []
        for rect in image_aug_rects:
            line = "bib,%i,%i,%i,%i" % (rect["min_x"], rect["min_y"], rect["max_x"], rect["max_y"])
            lines.append(line)
        return "\n".join(lines)

    def save_image(out_dir, image, image_identifier, kpts, augment_no = "", is_augmented = True):
        unique_id = "%s_%s%s" % (image_identifier, "aug" if is_augmented else "org", augment_no)
        print "Saving %s image '%s' as '%s'..." % ("augmented" if is_augmented else "original", image_identifier, unique_id)
        misc.imsave("%s/%s.jpg" % (out_dir, unique_id), image)
        with open("%s/%s.csv" % (out_dir, unique_id), "w") as csv:
            csv.write(generate_csv_for_image_kpts(image, image_identifier, kpts))

    ###
    # Augmentation begins
    ###

    seq = iaa.Sequential(
        [
            affine(),
            sometimes(one_of([add_pos(), add_neg()])),
            sometimes(one_of([mul_pos(), mul_neg()])),
            sometimes(blur(), 0.3)
        ],
        random_order=True
    )

    # Process (copy) all original data
    org_data = dict(zip(image_identifiers, zip(images, keypoints)))
    for image_identifier, data in org_data.items():
        img, kpts = data[0], data[1]
        save_image(out_dir, img, image_identifier, kpts, is_augmented = False)

    # Process (augment) all augmented data
    for i in range(AUGMENT_TIMES):
        # Process AUGMENT_TIMES images
        print "Augmentation Round %i/%i..." % (i + 1, AUGMENT_TIMES)
        seq_det = seq.to_deterministic()
        aug_images = seq_det.augment_images(images)
        aug_keypoints = seq_det.augment_keypoints(keypoints)
        aug_data = dict(zip(image_identifiers, zip(aug_images, aug_keypoints)))
        for image_identifier, data in aug_data.items():
            img, kpts = data[0], data[1]
            save_image(out_dir, img, image_identifier, kpts, augment_no = i)

def process(in_dir, out_dir):
    def photo_has_runners(label):
        # Returns true if the label has runners
        return len(label["TaggedRunners"]) > 0

    def load_image(filename, temp_files):
        # Must auto-orient (and scale) all images
        # Saves it to a temporary file that is deleted once done
        file_pointer, temp_file = mkstemp()
        temp_files.append((file_pointer, temp_file))
        print "Generating %s%% sampled version of '%s' to '%s'..." % (SCALE * 100, filename, temp_file)
        os.system("convert '%s' -auto-orient -resize %s%% '%s'" % (filename, int(SCALE * 100), temp_file))
        return misc.imread(temp_file)

    def clean_temp_files(temp_files):
        # Cleans all temporary files
        print "Cleaning tempfiles..."
        for (file_pointer, temp_path) in temp_files:
            print "Deleting tempfile %s..." % temp_path
            os.close(file_pointer)
            os.remove(temp_path)
        temp_files = []

    def extract_bib_keypoints_on_image_from_label(label):
        # Extracts bib keypoints from the data labels
        def extract_bib_keypoint_from_coords_str(coords_str):
            # Extracts scaled keypoints from the coords_str (i.e., "200, 300" => x=200, y=300)
            coords = [ int(int(pt) * SCALE) for pt in coords_str.split(', ') ]
            keypoint = ia.Keypoint(x=coords[0], y=coords[1])
            return keypoint

        def extract_bib_keypoints_from_runner(runner):
            # Extracts keypoints from specific runner
            coords = runner["Bib"]["PixelPoints"]
            return [ extract_bib_keypoint_from_coords_str(c) for c in coords ]

        # Extract the image
        image = images[image_identifiers.index(label["Identifier"])]

        # Flatten each runner down
        keypoints = np.array([ extract_bib_keypoints_from_runner(runner) for runner in label["TaggedRunners"] ]).flatten()

        # Return a single KeypointsOnImage
        return ia.KeypointsOnImage(keypoints, shape=image.shape)

    ###
    # Start of processing
    ###

    # Read in photos, their labels, then the image, then zip together
    glob_d = "%s/*.jpg" % in_dir
    all_files = glob(glob_d)

    # Split into 15%-sized batches
    n_batches =  int(len(all_files) / 15)
    batches = [all_files[i:i+n_batches] for i in range(0, len(all_files), n_batches)]

    print "Processing %i files in %i batches..." % (len(all_files), len(batches))

    for i, batch in enumerate(batches):
        print "Batch %i of %i (%i files)" % (i+1, len(batches), )

        # Declare temporary files which we will eventually clean up
        temp_files = []

        # Extract the labels for this batch
        labels = [json.load((open("%s.json" % p))) for p in batch if os.path.exists("%s.json" % p)]

        # Reject labels that are not tagged
        labels = [label for label in labels if photo_has_runners(label)]
        image_identifiers = [label["Identifier"] for label in labels]
        files_to_accept = tuple(["%s.jpg" % img_id for img_id in image_identifiers])
        image_files = [image_file for image_file in batch if image_file.endswith(files_to_accept)]

        # Load in the images
        images = [load_image(filename, temp_files) for filename in image_files]

        # Extract all bib sheets and their respective coordinates and map to scaled matrix
        keypoints = [ extract_bib_keypoints_on_image_from_label(label) for label in labels ]

        augment(images, image_identifiers, keypoints, out_dir)

        # Clean all temps when done!
        clean_temp_files(temp_files)

        print "End of batch %i of %i" % (i+1, len(batches))

if __name__ == "__main__":
    # Must provide in and out dir
    assert len(sys.argv) - 1 >= 2, "Must provide two arguments (in dir and out dir)"

    # Input directory
    in_dir = sys.argv[1]
    assert in_dir != None, "Missing input directory (argv[1])"

    # Setup output
    out_dir = sys.argv[2]
    assert out_dir != None, "Missing output directory (argv[2])"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process(in_dir, out_dir)
