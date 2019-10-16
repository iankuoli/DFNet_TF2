import pandas as pd
from tqdm import tqdm
from os.path import expanduser

from trainer.configuration import Config
from trainer.datasets import Dataset
from trainer.model import DFNet
from trainer.loss import InpaintLoss
from trainer.img_mask import mask_imgs
from trainer.plots import *

#
# Configuration Loading
# ----------------------------------------------------------------------------------------------------------------------

config = Config("config.yaml")


#
# Dataset Loading
# 1. *.flist is a file that comprises a list of urls each of which links to an image.
# 2. load images without masks, generating masks on-the-fly
# ----------------------------------------------------------------------------------------------------------------------
print("Data Loading ......")
imgs = Dataset(config.batch_size_train, config.batch_size_infer)

if config.data.data_flist.svhn[0].split(".")[1] == "flist":
    imgs.load_from_flist(expanduser(config.data.data_flist.svhn[0]),
                         expanduser(config.data.data_flist.svhn[1]))
elif config.data.data_flist.svhn[0].split(".")[1] == "mat":
    imgs.load_mat(expanduser(config.data.data_flist.svhn[0]),
                  expanduser(config.data.data_flist.svhn[1]))
print("Data loading is finished.")


#
# Create model
# ----------------------------------------------------------------------------------------------------------------------

optimizer = tf.keras.optimizers.Adam(1e-3)

# train the model

model = DFNet(en_ksize=config.model.en_ksize,
              de_ksize=config.model.de_ksize,
              blend_layers=config.model.blend_layers)
print("DFNet declaration is finished.")


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

loss_function = InpaintLoss(structure_layers=config.model.blend_layers, texture_layers=config.model.texture_layers,
                            w_l1=config.loss.w_l1, w_percep=config.loss.w_percep,
                            w_style=config.loss.w_style, w_tv=config.loss.w_tv)


def compute_gradients(targets, model):
    with tf.GradientTape() as tape:
        loss, loss_list = compute_loss(targets)
    return tape.gradient(loss, model.trainable_variables)


def compute_loss(targets):
    # image masking
    masked_imgs, mask = mask_imgs(targets, config.img_shape,
                                  config.mask.max_vertex, config.mask.max_angle,
                                  config.mask.max_length, config.mask.max_brush_width)

    masks = tf.tile(tf.expand_dims(mask, 0), [config.batch_size_train, 1, 1, 1])
    results, alphas, raws = model(masked_imgs, masks)
    loss, loss_list = loss_function(results, targets, masks)

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

    masked_imgs, mask = mask_imgs(example_data, config.img_shape,
                                  config.mask.max_vertex, config.mask.max_angle,
                                  config.mask.max_length, config.mask.max_brush_width)
    plot_reconstruction(model, example_data, masked_imgs, mask, example_data.shape[0])


#
# Show grid in 2D latent space
# ----------------------------------------------------------------------------------------------------------------------
# show_grid_2D(model)
