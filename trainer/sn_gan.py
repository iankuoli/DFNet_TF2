import tensorflow as tf
from tensorflow import keras
from sn_lib import SNConv, SNDense


class SN_Discriminator(keras.Model):

    def __init__(self, c_num):
        super(SN_Discriminator, self).__init__()
        self.c_num = c_num
        self.conv1 = SNConv(c_num, ksize=5, stride=2, name='conv1', training=True)
        self.conv2 = SNConv(c_num * 2, ksize=5, stride=2, name='conv2', training=True)
        self.conv3 = SNConv(c_num * 4, ksize=5, stride=2, name='conv3', training=True)
        self.conv4 = SNConv(c_num * 4, ksize=5, stride=2, name='conv4', training=True)
        self.conv5 = SNConv(c_num * 4, ksize=5, stride=2, name='conv5', training=True)
        self.conv6 = SNConv(c_num * 4, ksize=5, stride=2, name='conv6', training=True)
        self.dense = SNDense(1, name='dense', training=True)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dense(x)
        return x


class SNGAN(keras.Model):

    def __init__(self, gen, dis):
        """
        Initializer of the class
        :param gen: generator
        :param dis: discriminator
        """
        super(SNGAN, self).__init__()
        self.gen = gen
        self.dis = dis

    def call(self, target_imgs, masked_imgs, mask):
        inpainted_imgs, alphas, raws = self.gen(masked_imgs, mask)

        eval_pos = self.dis(target_imgs)
        eval_neg = self.dis(inpainted_imgs[0])

        return inpainted_imgs, alphas, raws, eval_pos, eval_neg

    @staticmethod
    def gan_loss(data_pos, data_neg):
        """
        sn_pgan loss function for GANs.
        - Wasserstein GAN: https://arxiv.org/abs/1701.07875
        """
        d_loss = tf.reduce_mean(tf.nn.relu(1 - data_pos)) + tf.reduce_mean(tf.nn.relu(1 + data_neg))
        g_loss = -tf.reduce_mean(data_neg)
        return g_loss, d_loss
