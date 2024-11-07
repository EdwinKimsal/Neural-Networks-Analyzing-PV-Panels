# Imports
from PIL import Image, ImageDraw
import json
import os
import sys

# Customizable variables
input_file = "Input JSON"
output_file = "Input PNG Images"
final_file_type = "png"
img_side = 5000

# Get this file's folder's directory
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

# Set file_paths
input_file_path = os.path.join(script_directory, input_file)
output_file_path = os.path.join(script_directory, output_file)

# Set file_list to being blank
file_list = []

# Set label_types equal to the different types of labels
label_types = ["pv", "notpv"]

# Create empty dict with each label
data = dict()

# Iterate through file_path for each file
for (root, dirs, file) in os.walk(input_file_path):
    for f in file:
        # Append file to file_list
        file_list.append(f)

# For each file in the input folder
for file in file_list:
    # Create file key
    data[file] = dict()

    # For each dict in types
    for type in label_types:
        # Add blank list of type to file key
        data[file][type] = []

# Iterate through dict for each file
for file in file_list:
    # Open file
    with open(os.path.join(input_file_path, file), "r") as f:
        # Load json file data
        d = json.load(f)

        # Iterate for each shape of file in json
        for key in d["shapes"]:
            # If key is a solar panel
            if key["label"] == "pv":
                # Append points
                data[file]["pv"].append(key["points"])

            # Else if key not a solar panel
            elif key["label"] == "notpv":
                # Append points
                data[file]["notpv"].append(key["points"])

# Iterate for each file in data dict
for f in data:
    # Form basic black image and add var for drawling
    mask = Image.new('RGB', (img_side, img_side), color='black')
    draw = ImageDraw.Draw(mask)

    # Iterate for each type (nv and notpv) in file in data dict
    for type in data[f]:
        # Iterate for each polygon in each type in file in data dict
        for poly in data[f][type]:
            # Convert polygon list into tuple for draw.polygon
            result = tuple(tuple(i) for x, i in enumerate(poly))

            # If the polygon is a solar panel
            if type == 'pv':
                # Draw a white polygon
                draw.polygon(result, fill="white")

            # If the polygon is NOT a solar panel
            elif type == 'notpv':
                # Draw a black polygon
                draw.polygon(result, fill="black")

    # Save mask png
    file = f"{f}.{final_file_type}"
    mask.save(os.path.join(output_file, file))
    print(file)

# # Print data dictionary
# print(data)