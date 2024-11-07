# Imports
import os
# import cv2

# Function to remove images
def remove(input_path, file):
    # Remove file
    os.remove(os.path.join(input_path, file))
    # Print file removed
    print("Removed: " + file)

    # Remove .json from file
    file = file.replace(".json", "")

    # Remove file
    os.remove(os.path.join(input_path, file))
    # Print file removed
    print("Removed: " + file)

# Main function
def main():
    # Path to Split Images, Output Images, and cmd
    input = "Output Split Images"
    output = "Output Split Images"
    cmd = os.getcwd()

    # Set paths
    input_path = os.path.join(cmd, input)
    output_path = os.path.join(cmd, output)

    # Set blank json list
    json_list = []

    # Iterate through all the files in Split Images
    for file in os.listdir(input_path):
        # If .json.png file, append to json_list
        if file.endswith(".json.png"):
            json_list.append(file)

    # Iterate through all the files in json_list
    for file in json_list:
        # If all pixels in file are black, remove file
        if cv2.imread(os.path.join(input_path, file)).all() == 0:
            remove(input_path, file)

        # Print file
        print(os.path.join(input_path, file))

# Call main function
main()