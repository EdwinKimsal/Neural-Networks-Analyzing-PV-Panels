import os
import numpy as np

import model_lib.file_management as fm
import model_lib.dataset as ds
import model_lib.augmentation as aug
from model_lib import evaluate
from model_lib import solar_model


import torch
torch.set_float32_matmul_precision('medium')

def main():
    ds_root = r'E:\datasets\PV Aerial\NY-labels'
    imgdir = os.path.join(ds_root, 'img_tiles')
    maskdir = os.path.join(ds_root, 'mask_tiles')

    out_root = r'E:\data\solar_rev'
    ds_filelist_dir = os.path.join(out_root, 'dataset_defns')

    seed = 2025
    channel_list = [0,1,2]
    cropsize = 576
    epochs = 10
    batch_size = 16

    load = True

    # Build the dataset definition files if they don't exist
    im_list = fm.read_file_list(os.path.join(ds_root, 'positive_tiles.txt'))
    fm.test_train_valid_split_list(im_list, im_list, ds_filelist_dir, n_set=1000, seed=seed, overwrite=True)


    # Get the lists of files
    test_img_list = fm.read_file_list(os.path.join(ds_filelist_dir, 'test_img_2025.txt'))
    test_mask_list = test_img_list.copy()
    train_img_list = fm.read_file_list(os.path.join(ds_filelist_dir, 'train_img_2025.txt'))
    train_mask_list = train_img_list.copy()
    valid_img_list = fm.read_file_list(os.path.join(ds_filelist_dir, 'valid_img_2025.txt'))
    valid_mask_list = valid_img_list.copy()

    # Add the full paths
    test_img_list = fm.append_pathnames(test_img_list, imgdir)
    train_img_list = fm.append_pathnames(train_img_list, imgdir)
    valid_img_list = fm.append_pathnames(valid_img_list, imgdir)
    test_mask_list = fm.append_pathnames(test_mask_list, maskdir)
    train_mask_list = fm.append_pathnames(train_mask_list, maskdir)
    valid_mask_list = fm.append_pathnames(valid_mask_list, maskdir)


    # Create them as datasets
    train_aug = aug.get_training_augmentation(cropsize)
    valid_aug = aug.get_scale_augmentation(cropsize)
    test_aug = aug.get_scale_augmentation(cropsize)
    train_ds = ds.Dataset(train_img_list, train_mask_list, train_aug, channel_list=channel_list)
    valid_ds = ds.Dataset(valid_img_list, valid_mask_list, valid_aug, channel_list=channel_list)
    test_ds = ds.Dataset(test_img_list, test_mask_list, test_aug, channel_list=channel_list)


    tmax = np.ceil(epochs * len(train_ds) / batch_size)
    # create model
    model = solar_model.SolarModel("FPN", "resnext50_32x4d", in_channels=3, t_max=tmax)

    if load and os.path.exists(r"D:\Code\Python\edwin-nn\Neural-Networks-Analyzing-PV-Panels\NN Library\lightning_logs\version_0\checkpoints"):
        import glob
        fn = glob.glob(os.path.join(r"D:\Code\Python\edwin-nn\Neural-Networks-Analyzing-PV-Panels\NN Library\lightning_logs\version_0\checkpoints", "*.ckpt"))[-1]
        model = solar_model.SolarModel.load_from_checkpoint(fn)
    else:
        solar_model.fit(model, train_ds, valid_ds, batch_size, epochs)

    evaluate.evaluate(model, valid_ds, test_ds, batch_size)

    evaluate.metrics_plot(r"D:\Code\Python\edwin-nn\Neural-Networks-Analyzing-PV-Panels\NN Library\lightning_logs\version_0\metrics.csv")

    evaluate.plot_test_frames(model, test_ds)


if __name__ == "__main__":

    main()