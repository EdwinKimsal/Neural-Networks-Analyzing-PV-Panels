import os
import numpy as np

def read_file_list(source_file, base_dir=None):
    """
    Read a list of strings from a file and return as a list.

    Parameters
    ----------
    source_file: str
        Full context of the file to read
    base_dir: str
        A root dir to append to each filename

    Returns
    -------
    list of strings
    """

    mylist = []
    with open(source_file, 'r') as f:
        for line in f.readlines():
            fix = line.replace("\n", "")
            if fix != "":
                if base_dir is not None:
                    fix = os.path.join(base_dir, fix)
                mylist.append(fix)
    return mylist

def test_train_valid_split_list(img_files, mask_files, output_root, n_set=None, test_train_valid=(0.2, 0.72, 0.08), seed=None, overwrite=False):
    """
    Split a list of image and mask filenames up into test_train_valid sets and save them
    to separate files.

    Parameters
    ----------
    img_files: list or tuple
        List of file basenames for images
    mask_files: list or tuple or None
        List of file basenames for masks.
    output_root: str
        Root path to directory where files will be output. Directory names
        coming out will be: test_img_SEED, test_mask_SEED, train_img_SEED,
        train_mask_SEED
    n_set: int (default None)
        Number of images in the total dataset. If None or 0, the whole list of
        images in the img_dir folder will be used.
    test_train_valid: list[float] (default [0.2, 0.72, 0.08])
        Fractions of test, train and valid sets relative to n_set.
    seed: int (default None)
        The seed to use to initialize np.random.seed. If None, ignore.
    overwrite: bool (default False)
        Should files be overwritten in the target destinations?

    Returns
    ----------
    Output files organized as a tuple as follows:
        (test_im_file, test_msk_file, train_im_file, train_msk_file, valid_im_file, valid_msk_file)
    """
    # Confirm that the lists are the same length
    if not len(mask_files) == len(img_files):
        raise ValueError("Image and Mask lists must be the same length.")

    # Make sure our output directories exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    test_im_file = os.path.join(output_root, f"test_img_{seed}.txt")
    test_msk_file = os.path.join(output_root, f"test_mask_{seed}.txt")
    train_im_file = os.path.join(output_root, f"train_img_{seed}.txt")
    train_msk_file = os.path.join(output_root, f"train_mask_{seed}.txt")
    valid_im_file = os.path.join(output_root, f"valid_img_{seed}.txt")
    valid_msk_file = os.path.join(output_root, f"valid_mask_{seed}.txt")

    # Test for outputs already existing
    for i_file in [test_im_file, test_msk_file,
                   train_im_file, train_msk_file,
                   valid_im_file, valid_msk_file]:
        if os.path.exists(i_file):
            if not overwrite:
                print(f"Output file {i_file} exists. Skipping operation.")
                return test_im_file, test_msk_file, \
                    train_im_file, train_msk_file, \
                    valid_im_file, valid_msk_file
            else:  # Overwrite
                os.remove(i_file)

    # Set seed if exists
    if seed is not None:
        np.random.seed(seed)

    # Downselect a subset by choosing indices
    if n_set is None or n_set < 0:
        n_set = len(img_files)
    chosen_inds = list(np.random.choice(range(len(img_files)), n_set, replace=False))
    chosen_inds.sort()

    # Calculate how many belong in each split
    ntest = int(test_train_valid[0] * n_set)
    ntrain = int(test_train_valid[1] * n_set)
    nvalid = int(test_train_valid[2] * n_set)

    # Coerce to match n_set if they don't
    if ntest + ntrain + nvalid != n_set:
        print("Set does not split evenly, biasing towards train.")
        ntrain = n_set - ntest - nvalid

    # Choose indices to belong to each category.
    # Remove extras from the list.
    chosen_inds_cp = chosen_inds.copy()
    test = list(np.random.choice(chosen_inds_cp, ntest, replace=False))
    for item in test:
        chosen_inds_cp.remove(item)
    # Choose Train Files
    train = list(np.random.choice(chosen_inds_cp, ntrain, replace=False))
    for item in train:
        chosen_inds_cp.remove(item)
    # Valid remains
    assert len(chosen_inds_cp) == nvalid
    valid = chosen_inds_cp

    with open(test_im_file, "w") as test_im, \
            open(test_msk_file, "w") as test_msk, \
            open(train_im_file, "w") as train_im, \
            open(train_msk_file, "w") as train_msk, \
            open(valid_im_file, "w") as valid_im, \
            open(valid_msk_file, "w") as valid_msk:

        for ind in chosen_inds:
            im_file = img_files[ind]
            msk_file = mask_files[ind]

            # Store to file depending on name
            if ind in test:
                test_im.write(im_file+"\n")
                test_msk.write(msk_file+"\n")
            elif ind in train:
                train_im.write(im_file+"\n")
                train_msk.write(msk_file+"\n")
            else:  # it's in valid
                valid_im.write(im_file+"\n")
                valid_msk.write(msk_file+"\n")

    return test_im_file, test_msk_file, \
        train_im_file, train_msk_file, \
        valid_im_file, valid_msk_file


def append_pathnames(basename_list, root_dir):
    """
    Append a root directory to a list of basenames.

    Parameters
    ----------
    basename_list: list
        List of basenames to append to the root directory
    root_dir: str
        Directory to append to the basenames

    Returns
    -------
    list
    """
    return [os.path.join(root_dir, item) for item in basename_list]