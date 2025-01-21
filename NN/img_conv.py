"""
Module used to convert four channel images to
three channel images to find the affect
of each individual channel (including
IR or Alpha) and affect of all four channels
combined
"""

# Import(s)
import numpy as np

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

    # Make a copy of arr
    arr_copy = arr.copy()

    # Iterate through each pixel in img
    for row in range(len(arr_copy)):
        for pixel in range(len(arr_copy[row])):
            # Find alpha channel
            a = arr_copy[row][pixel][3]

            # Calculate opacity multiplier
            opacity_mult = (255 - a) / 255

            # Iterate through RGB values in pixel
            for channel in range(3):
                arr_copy[row][pixel][channel] *=  opacity_mult

    # Deletes channel from arr
    arr = np.delete(arr_copy, 3, axis=2)

    # Return arr
    return arr