import albumentations as A


def _get_scale_transform_list(imsize, scale_mode="crop"):
    """
    Get the list of transformations to apply to scale the image to the correct size

    :param imsize: The size of the crop to apply
    :param scale_mode: The mode of the crop, either "crop" or "resize"

    :return: A list of Albumentations transformations
    """

    if scale_mode == "crop":
        # Crop deterministically here, rather than randomly
        scale_transform = [
            A.PadIfNeeded(imsize, imsize, p=1),
            A.Crop(x_min=0, y_min=0, x_max=imsize, y_max=imsize, p=1),
        ]
    elif scale_mode == "resize":
        # We could do the image scaling by resizing instead of cropping if we wanted
        scale_transform = [
            A.Resize(imsize, imsize, p=1)
        ]
    else:
        raise ValueError("Invalid scale_mode: must be 'crop' or 'resize'")

    return scale_transform


def get_scale_augmentation(imsize, scale_mode="crop"):
    """
    Define augmentations to apply to the validation dataset, which should be minimal
    Simply pad and crop to ensure that the size is correct

    :param imsize: The size of the crop to apply
    :param scale_mode: The mode of the crop, either "crop" or "resize"

    :return: An Albumentations Compose object
    """

    return A.Compose(_get_scale_transform_list(imsize, scale_mode))


def get_training_augmentation(imsize, scale_mode="crop", include_4channel=False):
    """
    Define augmentations to apply to the training dataset

    :return: An Albumentations Compose object
    """

    scale_transform = _get_scale_transform_list(imsize, scale_mode)

    train_transform = [

        # Flip images horizontally
        A.HorizontalFlip(p=0.5),  # horizontal flip, probability 50%

        # Scale by up to 20%, rotate by up to 30deg, shift by up to 10%
        # Do this to every image, matches usage in previous study
        # A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),
        A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), translate_percent=(-0.1, 0.1), p=1),

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
    ]

    # Add in the scaling transform
    train_transform = scale_transform + train_transform

    # Add in some channel sensitive augmentations
    if include_4channel:
        channel_sensitive = [
            # Commented out due to four channel imgs not being able to handle this augmentation
            # Apply some random sharpening or blurring, probability 90%
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
        train_transform += channel_sensitive

    return A.Compose(train_transform)
