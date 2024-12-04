""""
This script creates a list of all files in the
"JP2000_PNG Files" folder and writes them to a 
text file called jp2_list.txt.
"""

# Import(s)
import os

# Main Function
def main():
    # Set file names
    in_file = os.path.join(".", "JP2000_PNG Files")
    out_file = os.path.join(".", "NY-Q", "tiles", "jp2_list.txt")

    # Create a list of all files in "JP2000_PNG Files"
    files = os.listdir(in_file)

    # Write starting index
    with open(out_file, "w") as f:
        # Write each file to jp2_list.txt
        for file in files:
            f.write(f"{file}\n")


# Call main function
main()

# Signify end of program
print("\nEnd of Program")
