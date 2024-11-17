# Imports
from PIL import Image
import os

# Node function
def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root
    if l < n and arr[i] < arr[l]:
        largest = l

    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r

    # Change root, if needed
    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])  # swap

        # Heapify the root.

        heapify(arr, n, largest)


# The main function to sort an array of given size
def heapSort(arr):
    n = len(arr)

    # Build a maxheap.
    # Since last parent will be at (n//2) we can start at that location.
    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])  # swap
        heapify(arr, i, 0)


# Function to create datasets
def create_datasets(input_dir, png_files, mask_files, png_output, mask_output):
    # Print what it is working on
    print(f"Working On: {png_output.split("\\")[-1]}")

    # Save each file in png_files list
    for png_file in png_files:
        in_file = os.path.join(input_dir, png_file)
        out_file = os.path.join(png_output, png_file)
        with open(out_file, "w") as f:
            im = Image.open(in_file)
            im.save(out_file)

    # Save each file in mask_files list
    for mask_file in mask_files:
        in_file = os.path.join(input_dir, mask_file)
        out_file = os.path.join(mask_output, mask_file)
        with open(out_file, "w") as f:
            im = Image.open(in_file)
            im.save(out_file)


# Separate function
def separate(input_path):
    # Set list for .png and .json.png files
    png_files = []
    json_png_files = []

    # Iterate through each file in input_path
    for file in os.listdir(input_path):
        # Append to proper list (png_files or json_png_files)
        if "_mask_" in file:
            json_png_files.append(file)
        elif "_img_" in file:
            png_files.append(file)

    # Return tuple of lists
    return (png_files, json_png_files)


# Main function
def main():
    # Percent vars (all must add up to one)
    per_test = .15
    per_train = .75
    # per_val = Remainder

    # Get directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Set Dataset folder
    dataset_folder = "Datasets"

    # Input and Output File
    input_file = "Final Output Split Images"
    test_file = "Test Set"
    test_mask_file = "Test Mask Set"
    train_file = "Train Set"
    train_mask_file = "Train Mask Set"
    val_file = "Val Set"
    val_mask_file = "Val Mask Set"

    # Set file_paths
    input_path = os.path.join(script_directory, input_file)
    test_path = os.path.join(script_directory, dataset_folder, test_file)
    test_mask_path = os.path.join(script_directory, dataset_folder, test_mask_file)
    train_path = os.path.join(script_directory, dataset_folder, train_file)
    train_mask_path = os.path.join(script_directory, dataset_folder, train_mask_file)
    val_path = os.path.join(script_directory, dataset_folder, val_file)
    val_mask_path = os.path.join(script_directory, dataset_folder, val_mask_file)

    # Call separate function
    png_files, json_png_files = separate(input_path)

    # Order the lists
    heapSort(png_files)
    heapSort(json_png_files)

    # Get amount of files in each category
    num_test = int(len(png_files) * per_test)
    num_train = int(len(png_files) * per_train)

    # Create dataset lists
    test = png_files[:num_test]
    test_mask = json_png_files[:num_test]
    train = png_files[num_test:num_test + num_train]
    train_mask = json_png_files[num_test:num_test + num_train]
    val = png_files[num_test + num_train:]
    val_mask = json_png_files[num_test + num_train:]

    # Create dataset folders
    create_datasets(input_path, test, test_mask, test_path, test_mask_path)
    create_datasets(input_path, train, train_mask, train_path, train_mask_path)
    create_datasets(input_path, val, val_mask, val_path, val_mask_path)


# Call main function
main()