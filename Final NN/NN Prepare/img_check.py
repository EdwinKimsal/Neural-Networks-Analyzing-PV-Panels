# Import(s)
import os

# Main Function
def main():
    # File paths
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "JP2000_PNG Files")

    # Get all files in file_path
    file_list = os.listdir(file_path)

    # Iterate through each file in file_list
    for file in file_list:
        # Get byte size
        byte_size = os.path.getsize(os.path.join(file_path, file))

        # Call remove function if byte size is 97134
        if byte_size == 97134: # 97134 bytes means there is no data in the file
            os.remove(os.path.join(file_path, file))
            print(f"{file} has been removed")


# Call main function
main()

# Signify end of program
print("\nEnd of Program")