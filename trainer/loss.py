import tensorflow as tf
from tensorflow import keras

from trainer.utils import resize_like


class ReconstructionLoss(keras.Model):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.l1 = keras.losses.MAE

    def call(self, results, targets):
        loss = 0.
        for i, (res, target) in enumerate(zip(results, targets)):
            loss += tf.reduce_mean(self.l1(res, target))
        return loss / len(results)


class VGGFeature(keras.Model):

    def __init__(self):
        super(VGGFeature, self).__init__()

        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        outputs = [vgg.get_layer(name).output for name in layer_names]
        self.style_extractor = tf.keras.Model([vgg.input], outputs)

    def call(self, x):

        return self.style_extractor(x)


class PerceptualLoss(keras.Model):

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.l1loss = keras.losses.MAE

    def call(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(
                zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                n, w, h, c = feat_res.shape
                loss += tf.reduce_mean(self.l1loss(tf.reshape(feat_res, shape=[n, w*h*c]),
                                                   tf.reshape(feat_target, shape=[n, w*h*c])))
        return loss / len(vgg_results)


class StyleLoss(keras.Model):

    def __init__(self):
        super(StyleLoss, self).__init__()

        self.l1loss = keras.losses.MAE

    def gram(self, feature):
        n, h, w, c = feature.shape
        feature = tf.reshape(feature, shape=[n, h * w, c])
        gram_mat = tf.scan(lambda a, x: tf.matmul(tf.transpose(x, perm=[1, 0]), x),
                           feature,
                           initializer=tf.zeros([c, c]))
        return gram_mat / (c * h * w)

    def call(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                n, h, w, c = feat_res.shape
                res_tmp = tf.reshape(self.gram(feat_res), shape=[n, c*c])
                target_tmp = tf.reshape(self.gram(feat_target), shape=[n, c*c])
                loss += tf.reduce_mean(self.l1loss(res_tmp, target_tmp))
        return loss / len(vgg_results)


class TotalVariationLoss(keras.Model):

    def __init__(self, c_img=3):
        super(TotalVariationLoss, self).__init__()
        self.c_img = c_img

        kernel = tf.constant([[0, 1, 0],
                              [1, -2, 0],
                              [0, 0, 0]], dtype=tf.float32)
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), 0)

        self.kernel = tf.concat([kernel] * c_img, axis=-1)

    def gradient(self, x):
        return tf.nn.conv2d(x, filters=self.kernel, strides=1, padding='SAME')

    def call(self, results, mask):
        loss = 0.
        for i, res in enumerate(results):
            grad = self.gradient(res) * resize_like(mask, res)
            loss += tf.reduce_mean(tf.math.abs(grad))
        return loss / len(results)


class InpaintLoss(keras.Model):

    def __init__(self, c_img=3, w_l1=6., w_percep=0.1, w_style=240., w_tv=0.1,
                 structure_layers=[0, 1, 2, 3, 4, 5], texture_layers=[0, 1, 2]):

        super(InpaintLoss, self).__init__()

        self.l_struct = structure_layers
        self.l_text = texture_layers

        self.w_l1 = w_l1
        self.w_percep = w_percep
        self.w_style = w_style
        self.w_tv = w_tv

        self.reconstruction_loss = ReconstructionLoss()

        self.vgg_feature = VGGFeature()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss(c_img)

    def call(self, results, target, mask):

        targets = [resize_like(target, res) for res in results]

        loss_struct = 0.
        loss_text = 0.
        loss_list = {}

        if len(self.l_struct) > 0:

            struct_r = [results[i] for i in self.l_struct]
            struct_t = [targets[i] for i in self.l_struct]

            loss_struct = self.reconstruction_loss(struct_r, struct_t) * self.w_l1

            loss_list['reconstruction_loss'] = loss_struct.numpy()

        if len(self.l_text) > 0:

            text_r = [targets[i] for i in self.l_text]
            text_t = [results[i] for i in self.l_text]

            vgg_r = [self.vgg_feature(f) for f in text_r]
            vgg_t = [self.vgg_feature(t) for t in text_t]

            loss_style = self.style_loss(vgg_r, vgg_t) * self.w_style
            loss_percep = self.perceptual_loss(vgg_r, vgg_t) * self.w_percep
            loss_tv = self.tv_loss(text_r, mask) * self.w_tv

            loss_text = loss_style + loss_percep + loss_tv
            loss_list.update({
                'perceptual_loss': loss_percep.numpy(),
                'style_loss': loss_style.numpy(),
                'total_variation_loss': loss_tv.numpy()
            })

        loss_total = loss_struct + loss_text

        return loss_total, loss_list
