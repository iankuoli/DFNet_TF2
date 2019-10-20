import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.io import loadmat
import urllib
import cv2
import os
from os.path import expanduser


# Convert to float32 and Normalize images value from [0, 255] to [0, 1].
def data_normalize(x_train, x_valid):
    x_train, x_valid = np.array(x_train, np.float32), np.array(x_valid, np.float32)
    x_train, x_valid = x_train / 255., x_valid / 255.
    return x_train, x_valid


class Dataset(object):

    def __init__(self, batch_size_train, batch_size_infer, use_prefetch=True):

        self.train_data = None
        self.valid_data = None
        self.batch_size_train = batch_size_train
        self.batch_size_infer = batch_size_infer
        self.use_prefetch = use_prefetch
        self.train_size = 0
        self.valid_size = 0
        self.input_dim = 0

    def load_from_dir_batch(self, train_flist_path, valid_flist_path, output_wh):

        train_list_ds = tf.data.Dataset.list_files(expanduser(os.path.join(train_flist_path, "*")))
        valid_list_ds = tf.data.Dataset.list_files(expanduser(os.path.join(valid_flist_path, "*")))

        def process_path(file_path):
            # load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            # convert the compressed string to a 3D uint8 tensor
            img = tf.image.decode_jpeg(img, channels=3)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            (w, h) = output_wh
            return tf.image.resize(img, [w, h])

        def prepare_for_training(ds, cache=True, batch_size=32, shuffle_buffer_size=1000):
            # This is a small dataset, only load it once, and keep it in memory.
            # Use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
            if cache:
                ds = ds.cache(cache) if isinstance(cache, str) else ds.cache()
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)
            # Repeat forever and set batch size
            ds = ds.repeat().batch(batch_size)
            # `prefetch` lets the dataset fetch batches in the background while the model is training.
            return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_imgs_ds = train_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        valid_imgs_ds = valid_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_ds = prepare_for_training(train_imgs_ds, batch_size=self.batch_size_train)
        valid_ds = prepare_for_training(valid_imgs_ds, batch_size=self.batch_size_infer)

        self.train_data = train_ds
        self.valid_data = valid_ds
        self.train_size = len(os.listdir(expanduser(train_flist_path)))
        self.valid_size = len(os.listdir(expanduser(valid_flist_path)))

    def load_from_flist(self, train_flist_path, valid_flist_path, output_wh, is_url=False):
        train_img_list = []
        valid_img_list = []

        def read(line, is_url):
            if is_url:
                f = urllib.request.urlopen(line.strip())
                return np.asarray(bytearray(f.read()), dtype="float32")
            else:
                return cv2.imread(os.path.expanduser(line.strip())).astype(np.float32)

        num_train_files = sum(1 for line in open(train_flist_path, 'r', encoding='UTF-8'))
        with open(train_flist_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, total=num_train_files):
                img = cv2.resize(read(line, is_url), output_wh, interpolation=cv2.INTER_LINEAR)
                train_img_list.append(img)

        num_valid_files = sum(1 for line in open(valid_flist_path, 'r', encoding='UTF-8'))
        with open(valid_flist_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, total=num_valid_files):
                img = cv2.resize(read(line, is_url), output_wh, interpolation=cv2.INTER_LINEAR)
                valid_img_list.append(img)

        self.train_size = len(train_img_list)
        self.valid_size = len(valid_img_list)
        self.input_dim = train_img_list[0].shape

        self.train_data = tf.data.Dataset.from_tensor_slices(train_img_list).repeat().shuffle(min(500, self.train_size))
        self.valid_data = tf.data.Dataset.from_tensor_slices(valid_img_list)
        self.pre_process()

    def load_mat(self, train_mat_path, valid_mat_path):

        train = loadmat(train_mat_path)
        valid = loadmat(valid_mat_path)

        # Change format
        x_train, y_train = self.change_format(train)
        x_valid, y_valid = self.change_format(valid)

        # x_train, x_valid = data_normalize(x_train, x_valid)
        # y_train, y_valid = tf.argmax(y_train, 1), tf.argmax(y_valid, 1)

        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices(x_train.astype(np.float32))

        self.train_data = train_data.repeat().shuffle(5000)
        self.valid_data = tf.data.Dataset.from_tensor_slices(x_valid.astype(np.float32))

        self.pre_process()

        self.train_size = x_train.shape[0]
        self.valid_size = x_valid.shape[0]
        self.input_dim = x_train.shape[1:]

    def pre_process(self):
        if self.batch_size_train > 1:
            self.train_data = self.train_data.batch(self.batch_size_train)
            self.valid_data = self.valid_data.batch(self.batch_size_infer)
        if self.use_prefetch:
            self.train_data = self.train_data.prefetch(1)
            self.valid_data = self.valid_data.prefetch(1)

    @staticmethod
    def change_format(mat):
        """
        Convert X: (HWCN) -> (NHWC) and Y: [1,...,10] -> one-hot
        """
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        y = np.eye(10)[y]
        return x, y
