import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plot_reconstruction(dataset, epoch, model, targets, masked_imgs, mask, nex=8, zm=2):

    masks = tf.tile(mask, [nex, 1, 1, 1])
    results, alphas, raws = model(masked_imgs, masks)

    fig, axs = plt.subplots(ncols=nex, nrows=5, figsize=(zm * nex, zm * 4))
    for axi, (dat, lab) in enumerate(zip([targets, masked_imgs, results[0], alphas[0], raws[0]],
                                         ["targets", "inputs", "results", "alphas", "raws"])):
        for ex in range(nex):
            tmp = dat[ex].numpy().squeeze() / 255.
            axs[axi, ex].imshow(tmp, cmap=plt.get_cmap('Greys'), vmin=0, vmax=1)
            axs[axi, ex].axes.get_xaxis().set_ticks([])
            axs[axi, ex].axes.get_yaxis().set_ticks([])
        axs[axi, 0].set_ylabel(lab)

    plt.savefig("figs/%s_%s.png" % (dataset, epoch))
    plt.show()


def show_grid_2D(model):
    # sample from grid
    nx = ny = 10
    mesh_grid = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))
    mesh_grid = np.array(mesh_grid).reshape(2, nx * ny).T
    x_grid = model.decode(mesh_grid)
    x_grid = x_grid.numpy().reshape(nx, ny, 28, 28, 1)
    # fill canvas
    canvas = np.zeros((nx * 28, ny * 28))
    for xi in range(nx):
        for yi in range(ny):
            canvas[xi * 28:xi * 28 + 28, yi * 28:yi * 28 + 28] = x_grid[xi, yi, :, :, :].squeeze()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(canvas, cmap=plt.get_cmap('Greys'))
    ax.axis('off')
