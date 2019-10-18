import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.io import loadmat
import urllib
import cv2
import os


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
