"""
Gets all .jp2 files from Four Channel ZIP Files
(located on OneDrive) and writes each file on a
new line in a .txt file

.txt file is used in jp2000_file_conv.py
script for converting each .jp2 file to a
.png file
"""

# Import(s)
import os

# add_file function
def add_file(file, txt_file):
    with open(txt_file, "a") as f:
        f.write(f"{file}\n")


# Main function
def main():
    # Files paths
    cwd = os.getcwd()
    input_files = os.path.join(cwd, "Four Channel ZIP Files")
    output_txt_file = os.path.join(cwd, "NY-Q", "tiles", "jp2_to_png.txt")

    # Reset output_txt_file
    with open(output_txt_file, "w") as f:
        pass

    # Iterate through file_path
    for (root, dirs, file) in os.walk(input_files):

        # Iterate through each file
        for f in file:
            # If the file ends with .jp2 add it to output txt file
            if f.endswith(".jp2") == True:
                print(f)
                add_file(f, output_txt_file)

# Call main function
main()