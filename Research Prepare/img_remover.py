# Imports
import os
import cv2

# Function to remove images
def add(input_path, output_path, file_mask, file_img):
    # Add file
    os.save(os.path.join(output_path, file_mask))
    # Print file removed
    print("Added: " + file_mask)

    # Remove file
    os.save(os.path.join(output_path, file_img))

    # Print file removed
    print("Added: " + file_img)


# Main function
def main():
    # Path to Split Images, Output Images, and cmd
    input = "Output Split Images"
    output = "Final Output Split Images"
    cmd = os.getcwd()

    # Set paths
    input_path = os.path.join(cmd, input)
    output_path = os.path.join(cmd, output)

    # Set blank json list
    mask_list = []

    # Set blank image list
    img_list = []

    # Iterate through all the files in Split Images
    for file in os.listdir(input_path):
        # If .json.png file, append to json_list
        if "_mask_" in file:
            mask_list.append(file)

        else:
            img_list.append(file)

    # Iterate through all the files in json_list
    for i in range(len(mask_list)):
        # If all pixels in file are black, remove file
        if cv2.imread(os.path.join(input_path, mask_list[i])).all() != 0:
            add(input_path, output_path, mask_list[i], img_list[i])


# Call main function
main()