from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

import pandas as pd
import matplotlib.pyplot as plt

def evaluate(model, valid_ds, test_ds, batch):
    """
    Evaluate the model on the validation and test datasets printing results to the console
    """
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)

    valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False, num_workers=5, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=5, persistent_workers=True)

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)

def metrics_plot(fn):
    """
    Plot the training curves for the file
    :param fn:
    :return:
    """
    data = pd.read_csv(fn)
    data = data.groupby(data['epoch']).first()

    plt.plot(data.index, data['valid_per_image_iou'], label='Validation per image IoU')
    plt.plot(data.index, data['train_per_image_iou'], label='Training per image IoU')
    plt.legend()
    plt.show()

def plot_test_frames(model, test_ds, n=5):
    """
    Plot some sample test images
    :param model:
    :param test_ds:
    :param n:
    :return:
    """
    test_loader = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=5, persistent_workers=True)

    # Visualize results
    images, masks = next(iter(test_loader))
    with torch.no_grad():
        model.eval()
        logits = model(images)
    pr_masks = logits.sigmoid()
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        # Number of samples visualized
        if idx <= n+1:
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