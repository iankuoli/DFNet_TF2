from loss import *


def get_norm(name):
    if name == 'batch':
        norm = keras.layers.BatchNormalization()
    # elif name == 'instance':
    #   norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_activation(name):
    if name == 'relu':
        activation = keras.layers.ReLU()
    elif name == 'elu':
        activation = keras.layers.ELU()
    elif name == 'leaky_relu':
        activation = keras.layers.LeakyReLU(alpha=0.2)
    elif name == 'tanh':
        activation = keras.layers.Activation('tanh')
    elif name == 'sigmoid':
        activation = keras.layers.Activation('sigmoid')
    else:
        activation = None
    return activation


class Conv2dSame(keras.Model):

    def __init__(self, out_channels, kernel_size, stride):
        super(Conv2dSame, self).__init__()

        self.conv = keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding="same")


    @staticmethod
    def conv_same_pad(ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return left, right

    def call(self, x):
        return self.conv(x)


class ConvTranspose2dSame(keras.Model):

    def __init__(self, out_channels, kernel_size, stride):
        super(ConvTranspose2dSame, self).__init__()

        padding, output_padding = self.deconv_same_pad(kernel_size, stride)

        self.trans_conv = keras.layers.Conv2DTranspose(filters=out_channels, kernel_size=kernel_size, strides=stride,
                                                          padding=padding, output_padding=output_padding)

    @staticmethod
    def deconv_same_pad(self, ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def call(self, x):
        return self.trans_conv(x)


class UpBlock(keras.Model):

    def __init__(self, mode='nearest', scale=2, channel=None, kernel_size=4):
        super(UpBlock, self).__init__()

        self.mode = mode
        if mode == 'deconv':
            self.up = ConvTranspose2dSame(channel, kernel_size, stride=scale)
        else:
            self.up = keras.layers.UpSampling2D(size=scale, interpolation=mode)

    def call(self, x):
        return self.up(x)


class EncodeBlock(keras.Model):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super(EncodeBlock, self).__init__()

        self.c_in = in_channels
        self.c_out = out_channels

        layers = [Conv2dSame(self.c_out, kernel_size, stride)]

        if normalization:
            layers.append(get_norm(normalization))
        if activation:
            layers.append(get_activation(activation))
        self.encode = keras.Sequential(layers)

    def call(self, x):
        return self.encode(x)


class DecodeBlock(keras.Model):

    def __init__(self, c_from_up, c_from_down, c_out,
                 mode='nearest', kernel_size=4, scale=2, normalization='batch', activation='relu'):
        super(DecodeBlock, self).__init__()

        self.c_from_up = c_from_up
        self.c_from_down = c_from_down
        self.c_in = c_from_up + c_from_down
        self.c_out = c_out

        self.up = UpBlock(mode, scale, channel=c_from_up, kernel_size=scale)

        layers = []
        layers.append(Conv2dSame(self.c_out, kernel_size, stride=1))
        if normalization:
            layers.append(get_norm(normalization))
        if activation:
            layers.append(get_activation(activation))

        self.decode = keras.Sequential(layers)

    def call(self, x, concat=None):
        out = self.up(x)
        if self.c_from_down > 0:
            out = tf.concat([out, concat], axis=-1)   # TensorFlow is channel-last
        out = self.decode(out)
        return out


class BlendBlock(keras.Model):

    def __init__(self, c_in, c_out, ksize_mid=3, norm='batch', act='leaky_relu'):
        super(BlendBlock, self).__init__()
        c_mid = max(c_in // 2, 32)
        self.blend = keras.Sequential(
            [Conv2dSame(c_mid, 1, 1),
             get_norm(norm),
             get_activation(act),
             Conv2dSame(c_out, ksize_mid, 1),
             get_norm(norm),
             get_activation(act),
             Conv2dSame(c_out, 1, 1),
             get_activation('sigmoid')])

    def call(self, x):
        return self.blend(x)


class FusionBlock(keras.Model):

    def __init__(self, c_alpha=1):
        super(FusionBlock, self).__init__()
        c_img = 3
        self.map2img = keras.Sequential(
            [Conv2dSame(c_img, 1, 1),
             get_activation('sigmoid')])
        self.blend = BlendBlock(c_img*2, c_alpha)

    def call(self, img_miss, feat_de):
        img_miss = resize_like(img_miss, feat_de)
        raw = self.map2img(feat_de)
        alpha = self.blend(tf.concat([img_miss, raw], axis=-1))
        result = alpha * raw + (1 - alpha) * img_miss
        return result, alpha, raw


class DFNet(keras.Model):

    def __init__(self, c_img=3, c_mask=1, c_alpha=3,
                 mode='nearest', norm='batch', act_en='relu', act_de='leaky_relu',
                 en_ksize=[7, 5, 5, 3, 3, 3, 3, 3], de_ksize=[3]*8,
                 fuse_index=[0, 1, 2, 3, 4, 5]):
        super(DFNet, self).__init__()

        c_init = c_img + c_mask

        self.n_en = len(en_ksize)
        self.n_de = len(de_ksize)
        self.fuse_index = fuse_index

        assert self.n_en == self.n_de, 'The number layer of Encoder and Decoder must be equal.'
        assert self.n_en >= 1, 'The number layer of Encoder and Decoder must be greater than 1.'
        assert 0 in fuse_index, 'Layer 0 must be blended.'

        self.en = []
        c_in = c_init
        self.en.append(EncodeBlock(c_in, 64, en_ksize[0], 2, None, None))
        for k_en in en_ksize[1:]:
            c_in = self.en[-1].c_out
            c_out = min(c_in*2, 512)
            self.en.append(EncodeBlock(c_in, c_out, k_en, stride=2, normalization=norm, activation=act_en))

        # register parameters
        for i, en in enumerate(self.en):
            self.__setattr__('en_{}'.format(i), en)

        self.de = []
        self.fuse = []
        for i, k_de in enumerate(de_ksize):

            c_from_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            c_out = c_from_down = self.en[-i-1].c_in
            layer_idx = self.n_de - i - 1

            self.de.append(DecodeBlock(c_from_up, c_from_down, c_out, mode, k_de,
                                       scale=2, normalization=norm, activation=act_de))
            if layer_idx in fuse_index:
                self.fuse.append(FusionBlock(c_alpha))

    def call(self, masked_img, mask):

        out = tf.concat([masked_img, mask], axis=-1)

        out_en = [out]
        for encode in self.en:
            out = encode(out)
            out_en.append(out)

        results = []
        alphas = []
        raws = []
        
        for i, decode in enumerate(self.de):
            out = decode(out, out_en[-i - 2])
            map_to_fuse = self.n_de - 1 - i
            if map_to_fuse in self.fuse_index:
                fuse_index = self.fuse_index.index(map_to_fuse)
                fuse = self.fuse[fuse_index]
                result, alpha, raw = fuse(masked_img, out)
                results.append(result)
                alphas.append(alpha)
                raws.append(raw)

        return results[::-1], alphas[::-1], raws[::-1]
