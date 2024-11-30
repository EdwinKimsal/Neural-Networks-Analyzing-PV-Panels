# Imports
from PIL import Image
import os

# Customizable (global) vars
original_sub_img_side = 625  # original_sub_img_side must be equally divisible by this var)
modulo_num = 32
input_txt = "jp2_list.txt"
input_file = "JP2000_PNG Files"
output_file = "jp2"

# Get this file's folder's directory
cwd = os.getcwd()

# Set file_paths
input_txt_path = os.path.join(cwd, "NY-Q", "tiles", input_txt)
input_file_path = os.path.join(cwd, input_file)

# Final file type
final_file_type = "png"

# write_file Function
def write_file(file_list, img_per_row, final_sub_img_side):
    with open(input_txt_path, "w") as f:
        for line in file_list:
            f.write(f"{line}\n")

    return img_divider(file_list, img_per_row, final_sub_img_side)


# img_divider Function
def img_divider(file_list, img_per_row, final_sub_img_side):
    # Base case end
    if len(file_list) == 0:
        return file_list

    # Set file
    file = file_list[0]

    # Print file
    print(file)

    # Iterate for the img_y
    for i in range(img_per_row):

        # Iterate for the img_x
        for j in range(img_per_row):

            # Set file name
            file_name = f"{i*8+j}.{final_file_type}" # i*8+j is 0 to 63

            # Open image
            im = Image.open(os.path.join(input_file_path, file))

            # Crop THEN Resize (sub) img
            # .crop((left, top, right, bottom))
            im = im.crop((j * original_sub_img_side, i * original_sub_img_side, j * original_sub_img_side + original_sub_img_side, i * original_sub_img_side + original_sub_img_side))
            im = im.resize((final_sub_img_side, final_sub_img_side))

            # Save (sub) img as 8-bit
            # If ends with .json.png convert to 8 bit
            if file.endswith(".json.png"):
                im.convert('P').save(os.path.join(cwd, output_file, f"{file.replace(".json.png", "_mask")}_{file_name}"))

            # Save (sub) image
            # Else if ends with .png
            elif file.endswith(".png"):
                im.save(os.path.join(cwd, output_file, f"{file.replace(".png", "")}_{file_name}"))

    # Recall function removing first ele
    return write_file(file_list[1:], img_per_row, final_sub_img_side)


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
    return img_divider(file_list, img_per_row, final_sub_img_side)


# Start function
def main():
    # Create empty file_list
    file_list = []

    # Iterate through file_path
    with open(input_txt_path, "r") as f:
        for line in f:
            # Append file to file_list
            file_list.append(line.replace("\n", ""))

    # Print the number of files in file_list
    print(len(file_list))

    # Iterate through each file in file_list
    for file in file_list:

        # Check if the file is a png file
        im = Image.open(os.path.join(input_file_path, file))
        img_side = im.width
        break

    # Call calc_vars
    file_list = calc_vars(file_list, img_side)

    # Call Write File
    write_file(file_list)


# Call Start Function
main()

# State the program finished
print(f"\nEnd of Program")