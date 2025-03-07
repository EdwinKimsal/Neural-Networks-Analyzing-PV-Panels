# Set-Up Instructions

- Need dataset of images
- ```$ pip install segmentation-models-pytorch@git+https://github.com/EdwinKimsal/segmentation_models.pytorch```

## Note
- Model for training 1 (located on lines in the mid 400s on the Neural Networks) is for THREE channel images
- Model for training 2 (located on lines in the mid 400s on the Neural Networks) is for FOUR channel images
- Automatically set by default, but the neural networks are currently using different encoders for four and three channel images
- The Neural Networks for multichannel imgs can be found in the NN Library directory. The ```del_channel``` Neural Network is used to delete a channel from an RGBA img. The ```ir``` Neural Network is used for four channel RGBA imgs.

## Installing Pytorch
Use command from here to set up locally
https://pytorch.org/get-started/locally/

