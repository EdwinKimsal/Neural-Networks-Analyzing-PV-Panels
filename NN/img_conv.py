"""
Module used to convert four channel images to
three channel images to find the affect
of each individual channel (including
IR or Alpha) and affect of all four channels
combined
"""

# Import(s)
import numpy as np
import cv2

# Function to remove a channel from image
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


# Function to apply A part of RGBA to RGB (make four channel img a three channel img while saving data)
def apply_a(arr):
    """
    Applies the A to the RGB and removes the A-channel from the
    image. The core data stays the same, but the channels go from
    four to three

    Arguments
        - arr: Numpy array of RGBA img

    Return:
        - Numpy array of RGB img with A applied to each channel
    """

    # Extract the alpha channel
    alpha_channel = arr[:, :, 3] / 255

    # Blend the image with a white background using the alpha channel
    blended = cv2.convertScaleAbs(arr[:, :, :3] * alpha_channel[:, :, np.newaxis] + (1 - alpha_channel[:, :, np.newaxis]) * 255)

    # Return arr
    return blended