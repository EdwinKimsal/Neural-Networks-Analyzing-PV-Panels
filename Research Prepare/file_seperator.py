# Imports
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

# Customizable vars
input_file = "All Files"
output_file_png = "Input PNG Images"
output_file_json = "Input JSON"
output_file_jp2 = "Input JP2"

# Get this file's folder's directory
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

# Set file_paths
input_file_path = os.path.join(script_directory, input_file)
output_file_path_png = os.path.join(script_directory, output_file_png)
output_file_path_json = os.path.join(script_directory, output_file_json)

# Set file_list to blank
file_list = []

# Iterate through file_path
for (root, dirs, file) in os.walk(input_file_path):

    # Iterate through each file
    for f in file:

        # Append file to file_list
        file_list.append(f)

# Iterate through each file in the input folder
for file in file_list:

    # Print the file being worked on
    print(file)

    # If file is a png...
    if file.endswith(".png"):

        # Save PNG file to designated location
        im = Image.open(os.path.join(input_file_path, file))
        im.save(os.path.join(output_file_path_png, file))

    # If file is a JSON...
    elif file.endswith(".json"):

        # Save (dumb) JSON file in designated location
        with open(os.path.join(input_file_path, file), "r") as f:
            data = json.load(f)
            out_file = open(os.path.join(output_file_path_json, file), "w")
            json.dump(data, out_file)


    # If file is a JP2 (JPEG2000)
    elif file.endswith(".jp2"):

        # Save JP2 file in designated location
        img = np.asarray(Image.open(file))
        plt.imshow(img)
        plt.show()

        # Print file to user
        print(file)