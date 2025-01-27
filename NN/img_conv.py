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