"""
Module used to combine similar neural network code, into one
easy to use file to simplify debugging
"""

# Import(s)
import matplotlib.pyplot as plt
import numpy as np
import os
import albumentations as A

# Function(s)
def remove_channel(arr, channel=-1):
    """
    Converts 4-Channel numpy array to three channel
    numpy array

    Arguments:
        - arr: Numpy array of RGBA img
        - channel (default = -1): part of pixels to get rid of
            - R = 0, G = 1, B = 2, A = 3 (default)

    Return:
        - Numpy array of RGB img
    """

    # Deletes channel from arr
    arr = np.delete(arr, channel, axis=2)

    # Return arr
    return arr


def read_file_list(source_file, base_dir=None):
    """
    Read a list of strings from a file and return as a list.

    Parameters
    ----------
    source_file: str
        Full context of the file to read
    base_dir: str
        A root dir to append to each filename

    Returns
    -------
    list of strings
    """
    mylist = []
    with open(source_file, 'r') as f:
        for line in f.readlines():
            fix = line.replace("\n", "")
            if fix != "":
                if base_dir is not None:
                    fix = os.path.join(base_dir, fix)
                mylist.append(fix)
    return mylist


# helper function for data visualization
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image.transpose(1, 2, 0))  # Convert axes to display
    plt.show()


def get_training_augmentation(CROPSIZE):
    """
    Define augmentations to apply to the training dataset

    :return: An Albumentations Compose object
    """
    train_transform = [
        # # We could do the image scaling by resizing instead of cropping if we wanted
        # A.Resize(CROPSIZE, CROPSIZE, always_apply=True, p=1),

        # Flip images horizontally
        A.HorizontalFlip(p=0.5),  # horizontal flip, probability 50%

        # Scale by up to 20%, rotate by up to 30deg, shift by up to 10%
        # Do this to every image, matches usage in previous study
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),

        # Pad the image to make sure it's at least CROPSIZE x CROPSIZE
        # Then crop to CROPSIZE x CROPSIZE
        A.PadIfNeeded(min_height=CROPSIZE, min_width=CROPSIZE, always_apply=True),
        A.RandomCrop(height=CROPSIZE, width=CROPSIZE, always_apply=True),

        # Apply some random distortions in terms of noise and perspective
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        # Apply some random brightness and contrast adjustments, probability 90%
        A.OneOf(
            [
                # A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        # Apply some random sharpening or blurring, proability 90%
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        # Apply some random color adjustments, probability 90%
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(CROPSIZE):
    """
    Define augmentations to apply to the validation dataset, which should be minimal
    Simply pad and crop to ensure that the size is correct
    :return: An Albumentations Compose object
    """
    test_transform = [
        # # We could do the image scaling by resizing instead of cropping if we wanted
        # A.Resize(CROPSIZE, CROPSIZE, always_apply=True, p=1),

        # Crop deterministically here, rather than randomly
        A.PadIfNeeded(CROPSIZE, CROPSIZE, always_apply=True),
        A.Crop(x_min=0, y_min=0, x_max=CROPSIZE, y_max=CROPSIZE, always_apply=True),
    ]
    return A.Compose(test_transform)