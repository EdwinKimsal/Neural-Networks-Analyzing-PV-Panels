"""
Script creates two .txt files
- one for all split pngs in img dir
- one for all split jp2_pngs in jp2 dir
"""

# Import(s)
import os

# get_files function
def get_files(dir):
    """
    Function retrieves all files in a directory (dir)
    and returns them as a list of strings
    """

    # Set blank list for collection of files
    files = []

    # Iterate through each file in dir
    for (root, dirs, file) in os.walk(dir):
        # Iterate through each file
        for f in file:
            # Append f to files
            files.append(f)

    # Return list of all files
    return files


# save_file function
def save_file(files, save_dir):
    """
    Function saves all the files (in files list) in a
    directory (dir) as a txt file (located at save_dir)
    """

    # Open .txt file
    with open(save_dir, "w") as f:
        for file in files:
            f.write(file + "\n")


# Main function
def main():
    # Vars
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "NY-Q", "tiles")
    img_dir = os.path.join(data_dir, "img")
    save_dir = os.path.join(data_dir, "img_all.txt")

    # Call get_files function
    files = get_files(img_dir)

    # Call save_file to save files as .txt file
    save_file(files, save_dir)


# Call main function
main()