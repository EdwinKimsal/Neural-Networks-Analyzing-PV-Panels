"""
Uses img_all.txt and jp2_all.txt and creates
a set for both imgs and jp2s

Creates intersect_all.txt which is used to create seeding
"""

# Import(s)
import os

# get_sets function
def get_set(txt_path):
    """
    Returns a set called files of all files
    in a txt file, which is the parameter
    txt_path
    """

    # Set a blank set
    files = set()

    # Open txt_path to get set of all imgs
    with open(txt_path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            files.add(line)

    # Return set of files
    return files


# write_all function
def write_all(files, output_path):
    """
    Write intersection of set, parameter
    files, into a txt file, parameter
    outpath_file
    """

    # Open file to write in output_path
    with open(output_path, "w") as f:
        for file in files:
            f.write(f"{file}\n")


# Main function
def main():
    # Paths
    cwd =  os.getcwd()
    data_dir = os.path.join(cwd, "NY-Q", "tiles")
    img_txt_path = os.path.join(data_dir, "img_all.txt")
    jp2_txt_path = os.path.join(data_dir, "jp2_all.txt")
    output_path = os.path.join(data_dir, "all.txt")

    # Call get_sets function
    img_set = get_set(img_txt_path)
    jp2_set = get_set(jp2_txt_path)

    # Intersect two sets
    intersect_set = img_set.intersection(jp2_set)

    # Call write_all function
    write_all(intersect_set, output_path)

    # Print set sizes
    print(len(img_set))
    print(len(jp2_set))
    print(len(intersect_set))


# Call main function
main()