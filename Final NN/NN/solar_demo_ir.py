if __name__ == '__main__':
    import os
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    import torch

    import albumentations as A

    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as BaseDataset
    from torch.optim import lr_scheduler

    import segmentation_models_pytorch as smp
    import pytorch_lightning as pl
    from lightning import Trainer, LightningModule
    from pytorch_lightning.callbacks import ModelCheckpoint # For loading checkpoints

    # Root path to the dataset
    DATA_DIR = r'.\NY-Q\tiles'

    # File with the list of images to use for testing, training, and validation
    test = os.path.join(DATA_DIR, 'test_2024.txt')
    train = os.path.join(DATA_DIR, 'train_2024.txt')
    validate = os.path.join(DATA_DIR, 'val_2024.txt')

    # File for checkpoint
    check_point_file = os.path.join(".", 'lightning_logs', 'version_8', 'checkpoints', 'epoch=9-step=400.ckpt')

    # Size to crop the images during augmentation
    CROPSIZE = 576  # Must be divisible by 32

    # Some training hyperparameters
    BATCH_SIZE = 10
    EPOCHS = 2

    # Channel variables
    num_channels = 4
    type = "RGBA"

    # Paths to the images and masks in the dataset
    # Training
    x_train_dir = os.path.join(DATA_DIR, 'jp2')
    y_train_dir = os.path.join(DATA_DIR, 'mask')
    # Validation (same as training)
    x_valid_dir = x_train_dir
    y_valid_dir = y_train_dir
    # Test (Same as training)
    x_test_dir = x_train_dir
    y_test_dir = y_train_dir


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


    class Dataset(BaseDataset):
        """Dataset based on reading files, binary classes

        Args:
            images_dir (str): path to images folder
            masks_dir (str): path to segmentation masks folder
            imgs_fn (str): path to file listing names of train images
            masks_fn (str): path to file listing names of train masks
            augmentation (albumentations.Compose): data transfromation pipeline
                (e.g. flip, scale, etc.)

        """

        def __init__(
                self,
                images_dir,
                masks_dir,
                imgs_fn,
                masks_fn,
                augmentation=None,
        ):
            # Get a list of the filenames
            self.img_names = read_file_list(imgs_fn)
            self.mask_names = read_file_list(masks_fn)
            # Make sure that they're the same length
            assert len(self.img_names) == len(self.mask_names)

            # Append the directory path to the images and masks
            if images_dir is not None:
                self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.img_names]
            else:
                self.images_fps = self.img_names
            if masks_dir is not None:
                self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_names]
            else:
                self.masks_fps = self.mask_names

            # We only have one output class
            self.class_values = [1]

            # Store the augmentation function
            self.augmentation = augmentation

        def __getitem__(self, i):
            # Read the image and convert to img type
            image = Image.open(self.images_fps[i])
            image = image.convert(type)

            # Convert to numpy
            image = np.asarray(image, dtype=np.uint8)

            # Read the mask and convert to float32
            mask = Image.open(self.masks_fps[i])
            mask = np.asarray(mask, dtype=np.float32)
            mask = mask / np.max(mask)  # Normalize to range from 0 to 1

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # Transpose so dimensions match PyTorch's expectations
            # (sample, channels, height, width)
            return image.transpose(2,0,1), np.expand_dims(mask, axis=-1).transpose(2,0,1)

        def __len__(self):
            return len(self.img_names)

    # helper function for data visualization
    def visualize(**images):
        """Plot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image.transpose(1, 2, 0))  # Convert axes to display
        plt.show()

    # # Optional: Create a dataset with augmentation just to make sure it loads correctly
    # dataset = Dataset(
    #     x_train_dir,
    #     y_train_dir,
    #     train,
    #     train,
    #     augmentation=None
    # )
    # image, mask = dataset[0]
    # visualize(image=image, mask=mask, )


    def get_training_augmentation():
        """
        Define augmentations to apply to the training dataset

        :return: An Albumentations Compose object
        """
        train_transform = [
            # # We could do the image scaling by resizing instead of cropping if we wanted
            # A.Resize(CROPSIZE, CROPSIZE, always_apply=True, p=1),

            # Flip images horizontally
            A.HorizontalFlip(p=0.5),  # horizontal flip, probability 50%

            # Scale by up to 20%, rotate by up to 30deg, shift by up to 10%
            # Do this to every image, matches usage in previous study
            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),

            # Pad the image to make sure it's at least CROPSIZE x CROPSIZE
            # Then crop to CROPSIZE x CROPSIZE
            A.PadIfNeeded(min_height=CROPSIZE, min_width=CROPSIZE, always_apply=True),
            A.RandomCrop(height=CROPSIZE, width=CROPSIZE, always_apply=True),

            # Apply some random distortions in terms of noise and perspective
            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),

            # Apply some random brightness and contrast adjustments, probability 90%
            A.OneOf(
                [
                    # A.CLAHE(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            # Apply some random sharpening or blurring, proability 90%
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            # Apply some random color adjustments, probability 90%
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return A.Compose(train_transform)

    def get_validation_augmentation():
        """
        Define augmentations to apply to the validation dataset, which should be minimal
        Simply pad and crop to ensure that the size is correct
        :return: An Albumentations Compose object
        """
        test_transform = [
            # # We could do the image scaling by resizing instead of cropping if we wanted
            # A.Resize(CROPSIZE, CROPSIZE, always_apply=True, p=1),

            # Crop deterministically here, rather than randomly
            A.PadIfNeeded(CROPSIZE, CROPSIZE, always_apply=True),
            A.Crop(x_min=0, y_min=0, x_max=CROPSIZE, y_max=CROPSIZE, always_apply=True),
        ]
        return A.Compose(test_transform)

    # Generate a dataset with the augmentation just for example purposes
    augmented_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        train,
        train,
        augmentation=get_training_augmentation(),
    )

    # Should be same image with different random transforms, but comes out as different images
    for i in range(3):
        image, mask = augmented_dataset[3]
        visualize(image=image, mask=mask)

    # Generate the datasets for the training, including the augmentation
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        train,
        train,
        augmentation=get_training_augmentation(),
    )

    # Generate the datasets for validation and test, with only the cropping augmentation
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        validate,
        validate,
        augmentation=get_validation_augmentation(),
    )
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        test,
        test,
        augmentation=get_validation_augmentation(),
    )

    # Create the dataloaders, which will actually read the data off the disk
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Parameter for the model
    T_MAX = EPOCHS * len(train_loader)  # Actual max number of iterations
    OUT_CLASSES = 1

    class SolarModel(pl.LightningModule):
        """
        PyTorch Lightning module for Solar segmentation

        Args:
            arch: Architecture to use
            encoder_name: Name of the encoder to use
            in_channels: Number of input channels
            out_classes: Number of output classes
        """

        def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
            super().__init__()
            self.model = smp.create_model(
                arch,
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=out_classes,
                **kwargs
            )
            # preprocessing parameteres for image
            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view (1, num_channels, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, num_channels, 1, 1))

            # for image segmentation dice loss could be the best first choice
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

            # initialize step metics
            self.training_step_outputs = []
            self.validation_step_outputs = []
            self.test_step_outputs = []

        def forward(self, image):
            # normalize image here
            image = (image - self.mean) / self.std
            mask = self.model(image)
            return mask

        def shared_step(self, batch, stage):
            image, mask = batch

            # Shape of the image should be (batch_size, num_channels, height, width)
            # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
            assert image.ndim == 4

            # Check that image dimensions are divisible by 32,
            # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
            # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
            # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
            # and we will get an error trying to concat these features
            h, w = image.shape[2:]
            assert h % 32 == 0 and w % 32 == 0

            assert mask.ndim == 4

            # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
            assert mask.max() <= 1.0 and mask.min() >= 0

            logits_mask = self.forward(image)

            # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
            loss = self.loss_fn(logits_mask, mask)

            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        def shared_epoch_end(self, outputs, stage):
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])

            # per image IoU means that we first calculate IoU score for each image
            # and then compute mean over these scores
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

            # dataset IoU means that we aggregate intersection and union over whole dataset
            # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
            # in this particular case will not be much, however for dataset
            # with "empty" images (images without target class) a large gap could be observed.
            # Empty images influence a lot on per_image_iou and much less on dataset_iou.
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            metrics = {
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
            }

            self.log_dict(metrics, prog_bar=True)

        def training_step(self, batch, batch_idx):
            train_loss_info = self.shared_step(batch, "train")
            # append the metics of each step to the
            self.training_step_outputs.append(train_loss_info)
            return train_loss_info

        def on_train_epoch_end(self):
            self.shared_epoch_end(self.training_step_outputs, "train")
            # empty set output list
            self.training_step_outputs.clear()
            return

        def validation_step(self, batch, batch_idx):
            valid_loss_info = self.shared_step(batch, "valid")
            self.validation_step_outputs.append(valid_loss_info)
            return valid_loss_info

        def on_validation_epoch_end(self):
            self.shared_epoch_end(self.validation_step_outputs, "valid")
            self.validation_step_outputs.clear()
            return

        def test_step(self, batch, batch_idx):
            test_loss_info = self.shared_step(batch, "test")
            self.test_step_outputs.append(test_loss_info)
            return test_loss_info

        def on_test_epoch_end(self):
            self.shared_epoch_end(self.test_step_outputs, "test")
            # empty set output list
            self.test_step_outputs.clear()
            return

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }


    # Models
    # Model for Training
    # model = SolarModel("FPN", "resnext50_32x4d", in_channels=3, out_classes=OUT_CLASSES)
    # Model for ???
    # model = SolarModel("FPN", "mit_b0", in_channels=3, out_classes=OUT_CLASSES)
    # Model for Trained Checkpoint
    model = SolarModel.load_from_checkpoint(check_point_file, arch="FPN", encoder_name="resnext50_32x4d", in_channels=3,
                                            out_classes=OUT_CLASSES)
    # Load checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_logs/version_8/checkpoints/',
        filename=check_point_file,
        save_top_k=1,
        mode='max'
    )

    # Set trainer
    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

    # Trains Neural Network
    # trainer.fit(
    #     model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=valid_loader,
    # )

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)

    # Visualize results
    images, masks = next(iter(test_loader))
    with torch.no_grad():
        model.eval()
        logits = model(images)
    pr_masks = logits.sigmoid()
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        # Number of samples visualized
        if idx <= 4:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.numpy().squeeze())
            plt.title("Prediction")
            plt.axis("off")
            plt.show()
        else:
            break