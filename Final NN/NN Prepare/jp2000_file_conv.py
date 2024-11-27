# Import(s)
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

# Add file function
def convert(file, input_file_path, output_file_path):
    # Open JP2 image and convert to PNG
    img_jp2 = Image.open(os.path.join(input_file_path, file))
    img_jp2.save(os.path.join(output_file_path, f"{file.replace(".jp2", ".png")}"), "PNG")


# Main function
def main():
    # Files
    input_file = "Four Channel ZIP Files"
    output_file = "JP2000_PNG Files"

    # File paths
    cwd = os.getcwd()
    input_file_path = os.path.join(cwd, input_file)
    output_file_path = os.path.join(cwd, output_file)

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

        # If the file ends with .jp2 add it to output Folder
        if file.endswith(".jp2") == True:
            print(file)
            convert(file, input_file_path, output_file_path)


# Call main function
main()

# State the program Ended
print("End of program")