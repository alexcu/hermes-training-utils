# General Pipeline for Training

## 1. Augmentation

From a tagging iteration from [Argus](http://github.com/alexcu/argus), run batch
augmentation. Use `batch_augement.py` to do this:

```
$ python batch_augment.py /path/to/tagging/iteration /path/to/output [-t]
```

Providing the `-t` will show output of test images in the output directory.

## 2. Split Data for Training and Validation

All augmented data can now be split using the `split_train_val.py` script:

```
$ python split_train_val.py /path/to/augmented/images /path/to/output
```

If you have lots of training data, run this under `tmux`:

```
$ tmux new -s hermes_image_augmentation
```

and use `Ctrl+B D` to detactch from the session.

## 3. Train using Keras-FRCNN

Use the [Keras-FRCNN](https://github.com/alexcu/keras-frcnn) implementation
to train a model and write weights to disk to an hdf5 file.

You will need to set up Tensorflow to run this:

```
$ pip install tensorflow-gpu
```

You will need to [download CUDA](https://developer.nvidia.com/cuda-downloads)
for your platform.

To run Tensorflow on the GPU, download the cuDNN v5.1 Library for Linux.
This tar can be downloaded at [here](https://developer.nvidia.com/rdp/cudnn-download)
but you will need developer membership to access it.

![Download](https://i.imgur.com/zaBTptB.png)

Extract these to the relevant directories where CUDA is installed.

Be sure that the cuDNN library is on the LD path, e.g.:

```
$ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
```

Now run the training script (overnight). Be sure to run this under `tmux`.

The default network will be `resnet50`.

You should `tee` the output to a file, which will we use later. Replace [DATE]
with the relevant date. Use `--input_weight_path` to provide an input an input
`hdf5` if you have one.

We usually limit the number of epochs to 150, as we typically plateau here.

```
$ cd /path/to/keras-frcnn
$ python train_frcnn.py --path /path/to/split/data/result_train.txt \
                        --config_filename /path/to/out/models/[DATE]/config.pickle \
                        --output_weight_path /path/to/out/models/[DATE]/model_[DATE]_frcnn.hdf5 \
                        [--input_weight_path /path/to/out/models/[DATE]/model_[DATE]_frcnn.hdf5] \
                        --num_epochs 150 \
                        --parser simple | tee /path/to/log/training/[DATE].txt \
```

## 4. Scrape the log file

Scrape the log file from the training above for further analysis on how long
training took.

```
$ ./scrape_log.rb /path/to/log/training/[DATE].txt > /path/to/log/training/[DATE].csv
```

## 5. Validate the model

Under [hermes-bib-detect](https://github.com/alexcu/hermes-bib-detect), use
the `bib_detect.py` script to test the model using the validation images.

```
$ cd /path/to/hermes-bib-detect
$ python bib_detect.py -c /path/to/out/models/[DATE]/config.pickle \
                       -i /path/to/split/data/val \
                       -o /path/to/out/validated/[DATE]
```
