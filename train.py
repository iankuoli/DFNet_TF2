import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from os.path import expanduser

from configuration import Config
from datasets import Dataset
from model import DFNet
from loss import InpaintLoss
from img_mask import mask_imgs
from plots import *

#
# Configuration Loading
# ----------------------------------------------------------------------------------------------------------------------

config = Config("config.yaml")


#
# Dataset Loading
# 1. *.flist is a file that comprises a list of urls each of which links to an image.
# 2. load images without masks, generating masks on-the-fly
# ----------------------------------------------------------------------------------------------------------------------
imgs = Dataset(config.batch_size_train, config.batch_size_infer)
imgs.load_from_flist(expanduser(config.data.data_flist.celeba[0]),
                     expanduser(config.data.data_flist.celeba[1]))


#
# Create model
# ----------------------------------------------------------------------------------------------------------------------

# the optimizer for the model
optimizer = tf.keras.optimizers.Adam(1e-3)

# train the model
model = DFNet()


#
# Model Training
# ----------------------------------------------------------------------------------------------------------------------

# exampled data for plotting results
example_data = next(iter(imgs.valid_data))

# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns=['loss', 'reconstruction_loss', 'perceptual_loss', 'style_loss', 'total_variation_loss'])


n_epochs = 50
num_itr_per_batch_train = int(imgs.train_size / config.batch_size_train)
num_itr_per_batch_valid = int(imgs.valid_size / config.batch_size_infer)

# exampled data for plotting results
example_data = next(iter(imgs.valid_data))

loss_function = InpaintLoss(w_l1=config.w_l1, w_percep=config.w_percep, w_style=config.w_style, w_tv=config.w_tv)


def compute_gradients(targets, model):
    with tf.GradientTape() as tape:
        loss, loss_list = compute_loss(targets)
    return tape.gradient(loss, model.trainable_variables)


def compute_loss(targets):
    # image masking
    masked_imgs, mask = mask_imgs(targets)

    loss = 0
    loss_list = {'reconstruction_loss': 0.,
                 'perceptual_loss': 0.,
                 'style_loss': 0.,
                 'total_variation_loss': 0.}

    for masked_img, target in zip(masked_imgs, targets):
        results, alphas, raws = model(masked_img, mask)
        single_loss, single_loss_list = loss_function(results, target)
        loss += single_loss
        loss_list['reconstruction_loss'] += single_loss_list['reconstruction_loss']
        loss_list['perceptual_loss'] += single_loss_list['perceptual_loss']
        loss_list['style_loss'] += single_loss_list['style_loss']
        loss_list['total_variation_loss'] += single_loss_list['total_variation_loss']

    # Calculate the mean
    loss /= config.batch_size
    loss_list['reconstruction_loss'] /= config.batch_size
    loss_list['perceptual_loss'] /= config.batch_size
    loss_list['style_loss'] /= config.batch_size
    loss_list['total_variation_loss'] /= config.batch_size

    return loss, loss_list


for epoch in range(n_epochs):

    # train
    for batch, targets in tqdm(zip(range(num_itr_per_batch_train), imgs.train_data), total=num_itr_per_batch_train):

        gradients = compute_gradients(targets, model)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # test on holdout
    loss_batch = []
    for batch, targets in tqdm(zip(range(num_itr_per_batch_valid), imgs.valid_data), total=num_itr_per_batch_valid):

        loss, loss_list = compute_loss(targets)

        loss_batch.append(np.array([loss,
                                    loss_list['reconstruction_loss'],
                                    loss_list['perceptual_loss'],
                                    loss_list['style_loss'],
                                    loss_list['total_variation_loss']]))

    losses.loc[len(losses)] = np.mean(loss_batch, axis=0)

    # plot results
    print(
        "Epoch: {} | recon_loss: {} | latent_loss: {}".format(
            epoch, losses.recon_loss.values[-1], losses.latent_loss.values[-1]
        )
    )

    masked_imgs, mask = mask_imgs(example_data)
    plot_reconstruction(model, example_data, masked_imgs, mask, example_data.shape[0])


#
# Show grid in 2D latent space
# ----------------------------------------------------------------------------------------------------------------------
# show_grid_2D(model)
