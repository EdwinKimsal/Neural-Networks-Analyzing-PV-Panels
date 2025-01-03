"""
This script is used to create a dataset for the NY-Q project.
It will separate the positive and negative tiles based on the
sum of the pixels in the mask. It will then randomly order
the positive tiles and create a test, train, and validation
dataset based on the total number of images and the percentage
of test and train images. The output will be three txt files:
test, train, and validation.
"""

# Import(s)
import numpy as np
import cv2
import os

# pos_neg_sep function
def pos_neg_sep(file_path_all, file_path_mask, sum_min, pos_file, neg_file):
    # Open pos and neg files to write
    pos = open(os.path.join(".", "NY-Q", "tiles", pos_file), "w")
    neg = open(os.path.join(".", "NY-Q", "tiles", neg_file), "w")

    # Iterate through each img
    with open(file_path_all, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            mask = os.path.join(file_path_mask, line)
            file = cv2.imread(mask)
            height, width, channels = file.shape

            # Sum of all pixels in mask
            sum = np.sum(file == 255)

            # Write to pos or neg file based on sum and sum_min
            if sum >= sum_min:
                pos.write(f"{line}\n")
                print(line)
            else:
                neg.write(f"{line}\n")

        # Close pos and neg files
        pos.close()
        neg.close()


# Randomly order function
def random_ord(num_imgs, pos_file):
    # Open pos file to read
    pos = open(os.path.join(".", "NY-Q", "tiles", pos_file), "r")

    # Set an empty list
    img_list = []

    # Iterate through each line in pos file and append it until num imgs is met
    for line in pos:
        if len(img_list) < num_imgs:
            img_list.append(line)
        else:
            break

    # Return all imgs in pos file
    return img_list


# Create dataset function
def create_dataset(img_list, num, out_file):
    # Open out_file to write
    with open(os.path.join(".", "NY-Q", "tiles", out_file), "w") as f:
        # Iterate through img_list for the designated amount of times
        for img in img_list[:num]:
            f.write(img)

    # Return partial img_list
    return img_list[num:]


# Data separation function
def data_calc(total_num, test_per, train_per):
    # Calculate amount of imgs
    test_num = int(total_num * test_per)
    train_num = int(total_num * train_per)

    return test_num, train_num


# Main function
def main():
    # File path(s)
    cwd = os.getcwd()
    file_path_all = os.path.join(cwd, "NY-Q", "tiles", "all.txt")
    file_path_mask = os.path.join(cwd, "NY-Q", "tiles", "mask")
    pos_file = "positive_tiles.txt"
    neg_file = "negative_tiles.txt"

    # Customizable vars
    sum_min = 10000 # Min pv pixels needed
    seed = 2024 # Seed value to replicate data
    test_per = .1 # Percentage of test imgs in decimal form
    train_per = .8 # Percentage of train imgs in decimal form
    total_num = 100 # Total num of imgs

    # Call pos_neg_sep function
    pos_neg_sep(file_path_all, file_path_mask, sum_min, pos_file, neg_file)

    # Call random_ord function
    img_list = random_ord(total_num, pos_file)
    print(len(img_list))

    # Call data_sep function
    test_num, train_num = data_calc(total_num, test_per, train_per)

    # Create each dataset with designated num of imgs
    img_list = create_dataset(img_list, test_num, os.path.join(f"test_{str(seed)}.txt"))
    img_list = create_dataset(img_list, train_num, os.path.join(f"train_{str(seed)}.txt"))
    create_dataset(img_list, total_num - test_num - train_num, os.path.join(f"val_{str(seed)}.txt"))


# Call main function
main()