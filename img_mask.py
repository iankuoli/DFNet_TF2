import tensorflow as tf
import numpy as np
import cv2


def random_ff_mask(config):
    """Generate a random free form mask with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.IMG_SHAPES
    h, w, c = img_shape

    def np_mask():
        mask = np.zeros((h, w))
        num_v = 8 + np.random.randint(config.MAX_VERTEX)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(config.MAXANGLE)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(config.MAXLENGTH)
                brush_w = 10 + np.random.randint(config.MAXBRUSHWIDTH)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape(mask.shape+(1,)).astype(np.float32)

    mask = tf.py_func(np_mask, [], tf.float32, stateful=False)
    mask.set_shape([1,] + [h, w] + [1,])

    return mask


def mask_imgs(imgs, config):
    """
    Generate a mask, which will be applied to all the images in one batch
    :param imgs: images in a batch
    :param config: configuration
    :return:
    """

    mask = random_ff_mask(config)
    masked_imgs = imgs * (1. - mask)
    return masked_imgs, mask




