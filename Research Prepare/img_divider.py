# Imports
from PIL import Image
import sys
import os
import cv2

# Customizable vars
original_sub_img_side = 625  # original_sub_img_side must be equally divisible by this var)
modulo_num = 32
input_file = "Input PNG Images"
output_file = "Output Split Images"
final_file_type = "png"

# Get this file's folder's directory
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

# Set file_paths
input_file_path = os.path.join(script_directory, input_file)
output_file_path = os.path.join(script_directory, output_file)

# img_divider Function
def img_divider(file_list, img_per_row, final_sub_img_side):

    # Iterate through each file in file_list
    for file in file_list:

        if file.endswith(".json.png"):

            # Print the img being worked on
            print(file)

            # Set new_path based on the designated output_file and the current img
            new_path = os.path.join(output_file, file)

            # Check if the folder already exists
            if not os.path.exists(new_path):
                os.makedirs(new_path)

            # Iterate for the img_y
            for i in range(img_per_row):

                # Iterate for the img_x
                for j in range(img_per_row):

                    # Set file name
                    file_name = f"{i}_{j}.{final_file_type}"

                    # Open image
                    im = Image.open(os.path.join(input_file_path, file))

                    # Crop THEN Resize (sub) img
                    # .crop((left, top, right, bottom))
                    im = im.crop((j * original_sub_img_side, i * original_sub_img_side, j * original_sub_img_side + original_sub_img_side, i * original_sub_img_side + original_sub_img_side))
                    im = im.resize((final_sub_img_side, final_sub_img_side))

                    # Save (sub) img
                    # If ends with .json.png convert to 8 bit
                    if file.endswith(".json.png"):
                        print(type(im))
                        im = cv2.convertScaleAbs(im, alpha=0.03)

                    # Store image
                    im.save(os.path.join(new_path, file_name))

# calculate_vars Function
def calc_vars(file_list, img_side):

    # Calculate imgs_per_row
    img_per_row = img_side // original_sub_img_side

    # Calculate final_sub_img
    # If no need to scale
    if (original_sub_img_side % modulo_num == 0):

        # Set side len of the final_sub_img
        final_sub_img_side = original_sub_img_side

    # Else...
    else:
        # Calculate added_pixels_needed
        added_pixels_needed = modulo_num - original_sub_img_side % modulo_num

        # Calculate the side len of the final_sub_img
        final_sub_img_side = original_sub_img_side + added_pixels_needed

    # Call img_divider Function
    img_divider(file_list, img_per_row, final_sub_img_side)

# Start function
def start():

    # Create empty file_list
    file_list = []

    # Iterate through file_path
    for (root, dirs, file) in os.walk(input_file_path):

        # Iterate through each file
        for f in file:

            # Append file to file_list
            file_list.append(f)

    # Iterate through each file in file_list
    for file in file_list:

        # Check if the file is a png file
        im = Image.open(os.path.join(input_file_path, file))
        img_side = im.width
        break

    # Call calc_vars
    calc_vars(file_list, img_side)

# Call Start Function
start()

# State the program finished
print(f"\nProgram finished.")