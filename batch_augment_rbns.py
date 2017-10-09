#!/usr/bin/env python

import os
import sys
import imgaug as ia
import numpy as np
import scipy
import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from imgaug import augmenters as iaa
from glob import glob
from tempfile import mkstemp
import cv2

def augment_imgs(data):
    def affine():
        print("- Invoking affine")
        # Affine transformation
        TRANSLATE_PCT_RANGE = 0.5
        ROTATION_RANGE = (-30,30)
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
        print("- Invoking add_neg")
        # Applies a negative to all channels
        return iaa.Add((-45, 0))

    def add_pos():
        print("- Invoking add_pos")
        # Applies a positive to all channels
        return iaa.Add((0, 45))

    def mul_neg():
        print("- Invoking mul_neg")
        # Multiples all channels by a negative factor
        return iaa.Multiply((-2, 1))

    def mul_pos():
        print("- Invoking mul_pos")
        # Multiples all channels by a postive factor
        return iaa.Multiply((1, 2))

    def blur():
        print("- Invoking blur")
        # Chooses one of three blur methods
        return one_of([
            iaa.GaussianBlur((0, 3.0)),
            iaa.AverageBlur(k=(2, 4)),
            iaa.MedianBlur(k=(3, 5)),
        ])

    def sharpen():
        print("- Invoking sharpen")
        return iaa.Sharpen(alpha=(0.5,1), lightness=(0.75,1.5))

    def invert():
        print("- Invoking invert")
        return iaa.Invert(p=1)

    def add_to_hue_and_sat():
        print("- Invoking add_to_hue_and_sat")
        return iaa.AddToHueAndSaturation((-20, 20))

    def scale():
        print("- Invoking scale")
        return iaa.Affine(scale = {"x": (0.8,1.2), "y": (0.8,1.2)}, mode = "edge")

    def piecewise_affine():
        print("- Invoking piecewise_affine")
        return iaa.PiecewiseAffine((0.0010, 0.0045))

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    def sometimes(aug, pct = 0.5):
        return iaa.Sometimes(pct, aug)

    def one_of(funcs):
        # Shortcut for iaa.OneOf
        return iaa.OneOf(funcs)

    def valid_keypoints(kpts, img_ptr):
        # Returns any keypoints that are outside the width/height of the image
        width = img_ptr.shape[1]
        height = img_ptr.shape[0]
        # Copy over the "valid" keypoints (assume all are valid)
        for k in kpts:
            # If hidden, remove this runner
            if k.x < 0 or k.x > width or k .y < 0 or k.y > height:
                # Remove from valid if hidden
                valid_keypoints = [k for k in valid_keypoints if k is not kpts]
                break
        # Whatever remains becomes the Bib click points for these runners
        return valid_keypoints

    # Augmentation sequence
    seq = iaa.Sequential(
        [
            affine(),
            sometimes(one_of([add_pos(), add_neg()])),
            sometimes(one_of([mul_pos(), mul_neg()])),
            sometimes(blur(), 0.3),
            sometimes(invert()),
            sometimes(add_to_hue_and_sat()),
            sometimes(scale(), 0.60),
            sometimes(piecewise_affine())
        ],
        random_order=True
    )
    seq_det = seq.to_deterministic()

    # Original image keypoints and the augmented ones
    img_ids = [img_id for img_id, _ in data.items()]
    aug_chars = {}

    img_kpts = {}
    aug_img_kpts = {}

    img_ptrs = [img_data["img_ptr"] for _, img_data in data.items()]
    
    try:
        aug_img_ptrs = dict(zip(img_ids, seq_det.augment_images(img_ptrs)))
    except AssertionError:
        return None

    def chars_to_kpts(chars):
        kpts = np.array([])
        for char in chars:
            kpts = np.append(kpts, ia.Keypoint(x=char["top_left"][0],  y=char["top_left"][1]))
            kpts = np.append(kpts, ia.Keypoint(x=char["top_right"][0], y=char["top_right"][1]))
            kpts = np.append(kpts, ia.Keypoint(x=char["btm_right"][0], y=char["btm_right"][1]))
            kpts = np.append(kpts, ia.Keypoint(x=char["btm_left"][0],  y=char["btm_left"][1]))
        return kpts.flatten()

    img_kpts = [ia.KeypointsOnImage(chars_to_kpts(img_data["chars"]), shape=img_data["img_ptr"].shape)
                for _, img_data in data.items()]
    aug_img_kpts = dict(zip(img_ids, seq_det.augment_keypoints(img_kpts)))

    for img_id, aug_img_kpt in aug_img_kpts.items():
        height, width, _ = aug_img_kpt.shape
        if img_id not in aug_chars:
            aug_chars[img_id] = []
        assert len(aug_img_kpt.keypoints) % 4 == 0, "Augmented keypoints must be divisible by 4"
        # We want a skip of four so we can do:
        # 0,1,2,3 .. 4,5,6,7 .. 8,9,10,11
        for i in range(0, len(aug_img_kpt.keypoints) - 1, 4):
            aug_kpt_1 = aug_img_kpt.keypoints[i]
            aug_kpt_2 = aug_img_kpt.keypoints[i + 1]
            aug_kpt_3 = aug_img_kpt.keypoints[i + 2]
            aug_kpt_4 = aug_img_kpt.keypoints[i + 3]
            # TODO: Extract what the character is for this...
            label = "char"
            # Need to sort these such that {x,y}1 is min and
            # that {x,y}2 is max
            min_x = min(aug_kpt_1.x, aug_kpt_2.x, aug_kpt_3.x, aug_kpt_4.x)
            min_y = min(aug_kpt_1.y, aug_kpt_2.y, aug_kpt_3.y, aug_kpt_4.y)
            max_x = max(aug_kpt_1.x, aug_kpt_2.x, aug_kpt_3.x, aug_kpt_4.x)
            max_y = max(aug_kpt_1.y, aug_kpt_2.y, aug_kpt_3.y, aug_kpt_4.y)
            # Remove any invalid chars (points off screen)
            if min_x < 0 or max_x > width or min_y < 0 or max_y > height:
                continue
            aug_chars[img_id].append({
                "x1": min_x,
                "y1": min_y,
                "x2": max_x,
                "y2": max_y,
                "char": label
            })
    return aug_img_ptrs, aug_chars

def save_image(out_dir, batch_name, img_ptr, img_id, chars, augment_no = None):
    unique_id = "%s_%s" % (img_id, "org" if augment_no is None else ("aug%s" % augment_no))
    img_path = "%s/%s.jpg" % (out_dir, unique_id)
    annotated_img_path = "%s/%s.annotated.jpg" % (out_dir, unique_id)
    annotated_img_ptr = img_ptr.copy()
    for char in chars:
        top_left = (char["x1"], char["y1"])
        btm_right = (char["x2"], char["y2"])
        cv2.rectangle(annotated_img_ptr, top_left, btm_right, (0,255,0), 1)
    # Write annotated and non-annotated
    cv2.imwrite(annotated_img_path, annotated_img_ptr)
    cv2.imwrite(img_path, img_ptr)
    with open("%s/%s.csv" % (out_dir, batch_name), "a+") as csv:
        for char in chars:
            csv.write(",".join([
                img_path,
                str(char["x1"]),
                str(char["y1"]),
                str(char["x2"]),
                str(char["y2"]),
                char["char"],
            ]))
            csv.write("\n")

def augment_batch(imgs, in_dir, out_dir, batch_name, num_times = 50):
    assert batch_name in ["imgs", "imgs_bw"]
    img_prefix = {
        "imgs": "img",
        "imgs_bw": "img_bw"
    }[batch_name]
    gt_data = {}
    for img_id, chars in imgs.items():
        print("Loading %s in batch %s..." % (img_id, batch_name))
        img_path = "%s/%s/%s%s.jpg" % (in_dir, batch_name, img_prefix, img_id)
        print("--> %s" % img_path)
        img_ptr = cv2.imread(img_path)
        gt_data[img_id] = {
            "img_ptr": img_ptr,
            "chars": chars
        }
    # Batch augment data num_times
    for i in range(num_times):
        print("Augmentation Round %i/%i..." % (i + 1, num_times))
        augment_data = augment_imgs(gt_data)
        if augment_data is None:
            print("Augmentation failure")
            continue
        augmented_img_ptrs, augmented_chars = augment_data
        for img_id, img_ptr in augmented_img_ptrs.items():
            augmented_chars_for_img = augmented_chars[img_id]
            if len(augmented_chars_for_img) > 0:
                save_image(out_dir, batch_name, img_ptr, img_id, augmented_chars_for_img, i)


def process(gt_file, out_dir, num_rounds):
    # Define all the characters we have
    imgs = {}
    with open(gt_file, "r") as gt:
        for line in gt:
            comps = line.rstrip().split(",")
            img_id = os.path.splitext(os.path.basename(comps[0]))[0].replace("img", "")
            if img_id not in imgs:
                imgs[img_id] = []
            x1 = int(comps[1])
            y1 = int(comps[2])
            x2 = int(comps[3])
            y2 = int(comps[4])
            imgs[img_id].append({
                "top_left":  (x1, y1),
                "top_right": (x2, y1),
                "btm_right": (x2, y2),
                "btm_left":  (x1, y2),
                "char": comps[5]
            })
    in_dir = os.path.dirname(gt_file)
    print("Batch augment imgs")
    augment_batch(imgs, in_dir, out_dir, "imgs", num_rounds)
    print("Batch augment imgs_bw")
    augment_batch(imgs, in_dir, out_dir, "imgs_bw", num_rounds)

if __name__ == "__main__":
    # Must provide in and out dir
    assert len(sys.argv) - 1 >= 3, "Must provide 3 arguments (gt file, out dir and num augments)"

    # Input directory
    gt_file = sys.argv[1]
    assert gt_file != None, "Missing gt_file (argv[1])"

    # Setup output
    out_dir = sys.argv[2]
    assert out_dir != None, "Missing output directory (argv[2])"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_rounds = int(sys.argv[3])
    assert num_rounds != None, "Missing number of rounds (argv[3])"

    process(gt_file, out_dir, num_rounds)
