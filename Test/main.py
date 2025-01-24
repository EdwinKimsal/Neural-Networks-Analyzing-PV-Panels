# Import(s)
from PIL import Image
import numpy as np
import cv2
import os

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
    alpha_channel = arr[:, :, 3] / 255.0

    # Blend the image with a white background using the alpha channel
    blended = cv2.convertScaleAbs(arr[:, :, :3] * alpha_channel[:, :, np.newaxis] + (1 - alpha_channel[:, :, np.newaxis]) * 255)
    print(blended.shape)

    # Return arr
    return blended


# Main Function
def main():
    # Directories
    cwd = os.getcwd()
    rgb = os.path.join(cwd, "imgs", "rgb")
    rgba = os.path.join(cwd, "imgs", "rgba")
    altered = os.path.join(cwd, "imgs", "altered")

    # Iterate though each img in rgba
    for filename in os.scandir(rgba):
        if filename.is_file():
            # Get file
            cdn = os.path.basename(filename)

            # Open file as np arr
            image = Image.open(filename)
            arr = np.array(image)

            # Convert img
            arr = apply_a(arr)

            # Create img from arr and save to altered
            data = Image.fromarray(arr)
            data.save(os.path.join(altered, cdn))


# Call main function
main()