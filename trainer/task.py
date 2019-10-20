import pandas as pd
from tqdm import tqdm
import os
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
config = Config(os.path.join(os.getcwd(), "config_local.yaml"))          # config for local env.
# config = Config(os.path.join(os.getcwd(), "config_cloud.yaml"))      # config for cloud env.


#
# Dataset Loading
# 1. *.flist is a file that comprises a list of urls each of which links to an image.
# 2. load images without masks, generating masks on-the-fly
# ----------------------------------------------------------------------------------------------------------------------
print("Loading Data from %s ......" % expanduser(config.data.data_flist[config.data.dataset][0]))
imgs = Dataset(config.batch_size_train, config.batch_size_infer)

if len(config.data.data_flist[config.data.dataset][0].split(".")) == 1:
    imgs.load_from_dir_batch(expanduser(config.data.data_flist[config.data.dataset][0]),
                             expanduser(config.data.data_flist[config.data.dataset][1]),
                             tuple(config.img_shape[:2]))
else:
    if config.data.data_flist[config.data.dataset][0].split(".")[1] == "flist":
        imgs.load_from_flist(expanduser(config.data.data_flist[config.data.dataset][0]),
                             expanduser(config.data.data_flist[config.data.dataset][1]),
                             tuple(config.img_shape[:2]),
                             is_url=config.is_url)
    elif config.data.data_flist[config.data.dataset][0].split(".")[1] == "mat":
        imgs.load_mat(expanduser(config.data.data_flist[config.data.dataset][0]),
                      expanduser(config.data.data_flist[config.data.dataset][1]))

print("Data loading is finished.")


#
# Create model
# ----------------------------------------------------------------------------------------------------------------------
if config.optimizer == 'Adam':
    optimizer = tf.keras.optimizers.Adam(config.optimizer.args.lr)
elif config.optimizer == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(config.optimizer.args.lr)
elif config.optimizer == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(config.optimizer.args.lr)
else:
    optimizer = tf.keras.optimizers.SGD(config.optimizer.args.lr)

# train the model

model = DFNet(en_ksize=config.model.en_ksize,
              de_ksize=config.model.de_ksize,
              fuse_index=config.model.blend_layers)
print("DFNet declaration is finished.")


#
# Model Training
# ----------------------------------------------------------------------------------------------------------------------
def show_batch(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(config.batch_size_infer):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.axis('off')


# exampled data for plotting results
example_data = next(iter(imgs.valid_data))
show_batch(example_data.numpy())
plt.show()

# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns=['loss', 'reconstruction_loss', 'perceptual_loss', 'style_loss', 'total_variation_loss'])


n_epochs = 50
num_itr_per_batch_train = int(imgs.train_size / config.batch_size_train)
num_itr_per_batch_valid = int(imgs.valid_size / config.batch_size_infer)

# exampled data for plotting results
example_data = next(iter(imgs.valid_data))

loss_function = InpaintLoss(structure_layers=config.model.blend_layers, texture_layers=config.loss.texture_layers,
                            w_l1=config.loss.w_l1, w_percep=config.loss.w_percep,
                            w_style=config.loss.w_style, w_tv=config.loss.w_tv)


def compute_gradients(targets, model):
    with tf.GradientTape() as tape:
        loss, loss_list = compute_loss(targets, batch_size=config.batch_size_train)
    return tape.gradient(loss, model.trainable_variables)


def compute_loss(targets, batch_size):
    # image masking
    masked_imgs, mask = mask_imgs(targets, config.img_shape,
                                  config.mask.max_vertex, config.mask.max_angle,
                                  config.mask.max_length, config.mask.max_brush_width)

    masks = tf.tile(mask, [batch_size, 1, 1, 1])
    results, alphas, raws = model(masked_imgs, masks)
    loss, loss_list = loss_function(results, targets, masks)

    return loss, loss_list


for epoch in range(n_epochs):

    # train
    for batch, targets in tqdm(zip(range(num_itr_per_batch_train), imgs.train_data), total=num_itr_per_batch_train):

        gradients = compute_gradients(targets, model)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if batch % 2 == 0:
            # test on holdout
            loss_batch = []
            for batch_i, targets_i in zip(range(num_itr_per_batch_valid), imgs.valid_data):

                if batch_i > 10:
                    break

                loss, loss_list = compute_loss(targets_i, batch_size=config.batch_size_infer)

                loss_batch.append(np.array([loss,
                                            loss_list['reconstruction_loss'],
                                            loss_list['perceptual_loss'],
                                            loss_list['style_loss'],
                                            loss_list['total_variation_loss']]))

            losses.loc[len(losses)] = np.mean(loss_batch, axis=0)

            # plot results
            print(
                "Epoch: {} | recon_loss: {} | perceptual_loss: {} | style_loss: {} | total_variation_loss: {}".format(
                    epoch, losses.reconstruction_loss.values[-1], losses.perceptual_loss.values[-1],
                    losses.style_loss.values[-1], losses.total_variation_loss.values[-1]
                )
            )

            masked_imgs, mask = mask_imgs(example_data, config.img_shape,
                                          config.mask.max_vertex, config.mask.max_angle,
                                          config.mask.max_length, config.mask.max_brush_width)
            plot_reconstruction(config.data.dataset, str(epoch), model, example_data, masked_imgs, mask,
                                nex=example_data.shape[0])

            # Save the model into ckpt file
            model.save_weights(config.model.save_path, save_format='tf')

    # test on holdout
    loss_batch = []
    for batch, targets in tqdm(zip(range(num_itr_per_batch_valid), imgs.valid_data), total=num_itr_per_batch_valid):

        loss, loss_list = compute_loss(targets, batch_size=config.batch_size_infer)

        loss_batch.append(np.array([loss,
                                    loss_list['reconstruction_loss'],
                                    loss_list['perceptual_loss'],
                                    loss_list['style_loss'],
                                    loss_list['total_variation_loss']]))

    losses.loc[len(losses)] = np.mean(loss_batch, axis=0)

    # plot results
    print(
        "Epoch: {} | recon_loss: {} | perceptual_loss: {} | style_loss: {} | total_variation_loss: {}".format(
            epoch, losses.reconstruction_loss.values[-1], losses.perceptual_loss.values[-1],
            losses.style_loss.values[-1], losses.total_variation_loss.values[-1]
        )
    )

    masked_imgs, mask = mask_imgs(example_data, config.img_shape,
                                  config.mask.max_vertex, config.mask.max_angle,
                                  config.mask.max_length, config.mask.max_brush_width)
    plot_reconstruction(config.data.dataset, str(epoch), model, example_data, masked_imgs, mask, nex=example_data.shape[0])

    # Save the model into ckpt file
    model.save_weights(config.model.save_path, save_format='tf')

