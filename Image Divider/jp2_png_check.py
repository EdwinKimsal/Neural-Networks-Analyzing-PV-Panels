# Imports
from PIL import Image
import numpy as np
import sys
import os

# Customizable vars
input_folder = "Input JP2"
output_folder = "Output JP2 To PNG"

# Set empty list for pixel data
pixel_list = []

# Get this file's folder's directory
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

# Set file_paths
input_file_path = os.path.join(script_directory, input_folder)
output_file_path = os.path.join(script_directory, output_folder)

# Create empty file_list
file_list = []

# Iterate through file_path
for (root, dirs, file) in os.walk(input_file_path):

    # Iterate through each file
    for f in file:

        # Append file to file_list
        file_list.append(f)

# Set empty list for bool_list
bool_list = []

# Iterate through file_list for each file
for file in file_list:

    # Open JP2 image and convert to numpy array
    img_JP2 = Image.open(os.path.join(input_file_path, "000135.jp2"))
    data_JP2 = np.asarray(img_JP2)

    # Open/Save PNG image and convert to numpy array
    img_PNG = img_JP2
    img_PNG.save(file, "PNG")
    data_PNG = np.asarray(img_PNG)

    # If the PNG and JP2 images are the same
    if (data_PNG == data_JP2).all():

        # Tell the user and append True to bool_list
        print(f"Both images are the same ({file})")
        print()
        bool_list.append(True)

    # Else the PNG and JP2 images are different
    else:

        # Tell the user and append False to bool_list
        print(f"Both images are different ({file})")
        print()
        bool_list.append(False)

# Check if any images are different and tell the user
if False in bool_list:
    print("Not all images are the same")

else:
    print("All images are the same")