# Set-Up Instructions

- Need dataset of images
- ```$ pip install segmentation-models-pytorch@git+https://github.com/EdwinKimsal/segmentation_models.pytorch```

## Note
- Model for training 1 (located on lines in the mid 400s on the Neural Networks) is for THREE channel images
- Model for training 2 (located on lines in the mid 400s on the Neural Networks) is for FOUR channel images
- Automatically set by default, but the neural networks are currently using different encoders for four and three channel images
