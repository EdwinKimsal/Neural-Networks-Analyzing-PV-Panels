"""
Module used to convert four channel images to
three channel images to find the affect
of each individual channel (including
IR or Alpha)
"""

# Import(s)
import numpy as np

# Function to convert image based on channels
def four_to_three(arr, channel=-1):
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
    img = np.delete(arr, channel, axis=2)

    # Return img
    return img