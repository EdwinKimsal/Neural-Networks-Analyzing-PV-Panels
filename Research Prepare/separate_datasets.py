# Imports
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


# # Function to create datasets
# def create_datasets(png_files, json_png_files):
#     # Calculate the number of files for each dataset
#     num_test = len(png_files) * per_test
#     num_train = len(png_files) * per_train
#     num_val = len(png_files) * per_val
#
#     # Create datasets
#     test_set = png_files[:num_test]
#     train_set = png_files[num_test:num_test + num_train]
#     val_set = png_files[num_test + num_train:]
#
#     # Return datasets
#     return test_set, train_set, val_set


# Separate function
def separtate(input_path):
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
    per_val = .10

    # Dataset lists
    test = []
    test_mask = []
    train = []
    train_mask = []
    val = []
    val_mask = []

    # Get directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Input and Output File
    input_file = "Output Split Images"
    output_file = "Output Split Images"

    # Set file_paths
    input_path = os.path.join(script_directory, input_file)
    output_path = os.path.join(script_directory, output_file)

    # Call separate function
    png_files, json_png_files = separtate(input_path)

    # Order the lists
    heapSort(png_files)
    heapSort(json_png_files)

    # Print each list
    print(png_files)
    print(json_png_files)


# Call main function
main()