# Import(s)
import numpy as np
import cv2
import os


# Main function
def main():
    # File paths
    file_path_img = "./NY-Q/tiles/img"
    file_path_mask = "./NY-Q/tiles/mask"

    # Customizable vars
    sum_min = 100 # Min pv pixels needed

    # Open pos and neg files
    pos = open(os.path.join(".", "NY-Q", "positive_tiles.txt"), "w")
    neg = open(os.path.join(".", "NY-Q", "negative_tiles.txt"), "w")

    # Iterate through each img
    for f in os.listdir(file_path_mask):
        file = os.path.join(file_path_mask, f)
        file = cv2.imread(file)
        height, width, channels = file.shape

        # Sum of all pixels in mask
        sum = np.sum(file[0:height, 0:width])

        # Write to pos or neg file based on sum and sum_min
        if sum >= sum_min:
            pos.write(f + "\n")
            print(f)
        else:
            neg.write(f + "\n")

    # Close pos and neg files
    pos.close()
    neg.close()


# Call main function
main()