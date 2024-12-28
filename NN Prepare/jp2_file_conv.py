"""
.txt file is created in jp2_png_conv_txt.py
script

This script will convert all JP2 files in the
input dir to PNG files in the output dir.
"""

# Import(s)
from PIL import Image
import glymur
import os

# get_list function
def get_list(input_txt_file):
    # Open input_txt_file
    with open(input_txt_file, "r") as f:
        # Set blank list to store files
        file_li = []

        # Iterate through all jp2 files in txt file
        for jp2_file in f:
            file_li.append(jp2_file)

    return file_li


# Add file function
def convert(input_txt_file, input_dir, output_dir):
    # Get file_li
    file_li = get_list(input_txt_file)

    # Base case
    if len(file_li) == 0:
        break

    # Recursive case
    else:
        # Set working file and remove working file from list
        working_file = file_li[0].replace("\n", "")
        file_li.pop(0)

        # Print working file
        print(working_file)

        # Convert jp2 to png and save
        # Open the JP2 file
        jp2 = glymur.Jp2k(os.path.join(input_dir, working_file))

        # Access the pixel data as a NumPy array
        np_array = jp2[:]

        # Save array as png
        image = Image.fromarray(np_array)

        # Save the image as a PNG file
        image.save(os.path.join(output_dir, working_file.replace("jp2", "png")))

        # Remove working file from txt file
        # Open txt file to write
        with open(input_txt_file, "w") as f:
            # Iterate through each file in file_li
            for jp2_file in file_li:
                # Add file to txt
                f.write(jp2_file)

        # Recursive call
        convert(input_txt_file, input_dir, output_dir)


# Main function
def main():
    # File paths
    cwd = os.getcwd()
    input_txt_file = os.path.join(cwd, "NY-Q", "tiles", "jp2_all.txt")
    input_dir = os.path.join(cwd, "Four Channel ZIP Files")
    output_dir = os.path.join(cwd, "JP2000_PNG Files")

    # Print num of files left
    print(len(get_list(input_txt_file)))

    # Get file to work on (initiate process)
    convert(input_txt_file, input_dir, output_dir)


# Call main function
main()

# State the program Ended
print("End of program")