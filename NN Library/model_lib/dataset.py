from PIL import Image
from torch.utils.data import Dataset as BaseDataset
import numpy as np


# Dataset class
class Dataset(BaseDataset):
    """Dataset based on reading files, binary classes

    Args:
        image_list (list): list of image files
        mask_list (list): list of mask files
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        channel_list (list): list of channels to extract from the image
    """

    # Init method
    def __init__(
            self,
            image_list,
            mask_list,
            augmentation=None,
            channel_list=[0,1,2,3]
    ):
        self.image_list = image_list
        self.mask_list = mask_list
        assert len(self.image_list) == len(self.mask_list)

        # We only have one output class
        self.class_values = [1]

        # Store the augmentation function
        self.augmentation = augmentation

        # Set channel
        self.channel_list = channel_list

    # Get item method
    def __getitem__(self, i):
        im_file = self.image_list[i]
        mask_file = self.mask_list[i]

        image = read_image(im_file, channel_list=self.channel_list)
        mask = read_mask(mask_file)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Transpose so dimensions match PyTorch's expectations
        # (sample, channels, height, width)
        image = image.transpose(2, 0, 1)
        mask = np.expand_dims(mask, axis=-1).transpose(2, 0, 1)

        return image, mask

    # Get length method
    def __len__(self):
        return len(self.image_list)


def read_image(im_file, channel_list=[0,1,2,3]):
    """
    Read an image file and return the image as a numpy array
    :param im_file:
        Full path filename for file
    :param channel_list:
        List of channels to extract from the image
    :return:
        Numpy array of the image
    """
    # Read the image
    image = Image.open(im_file)

    # Convert to numpy
    image = np.asarray(image, dtype=np.uint8)

    image = image[:, :, channel_list]

    return image


def read_mask(mask_file):
    """
    Read a mask file and return the mask as a numpy array
    :param mask_file:
        Full path filename for mask file
    :return:
        Numpy array of the mask ranging from 0 to 1
    """
    # Read the mask and convert to float32
    mask = Image.open(mask_file)
    mask = np.asarray(mask, dtype=np.float32)
    mask = mask / np.max(mask)  # Normalize to range from 0 to 1

    return mask

