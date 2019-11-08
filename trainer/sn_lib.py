import tensorflow as tf
import numpy as np
import warnings

NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False, name="sn_w"):
    # Usually num_iters = 1 will be enough
    w_shape = W.shape.as_list()
    w_reshaped = tf.reshape(W, [-1, w_shape[-1]])
    if u is None:
        u = tf.random.normal(shape=[1, w_shape[-1]], name="u")
    
    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(w_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, w_reshaped))
        return i + 1, u_ip1, v_ip1
    
    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32), 
                   u, 
                   tf.zeros(dtype=tf.float32, shape=[1, w_reshaped.shape.as_list()[0]])))
    
    if update_collection is None:
        warnings.warn('Setting update_collection to None will make u being updated every W execution. '
                      'This maybe undesirable. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, w_reshaped), tf.transpose(u_final))[0, 0]
        w_bar = w_reshaped / sigma

        u = u_final
        with tf.control_dependencies([u]):
            w_bar = tf.reshape(w_bar, w_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, w_reshaped), tf.transpose(u_final))[0, 0]
        w_bar = w_reshaped / sigma
        w_bar = tf.reshape(w_bar, w_shape)
        
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != NO_OPS:
            u = u_final
            tf.compat.v1.add_to_collection(update_collection, u)
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


class SNConv(tf.keras.layers.Layer):

    def __init__(self, cnum, ksize, stride=1, rate=1, name='conv',
                 padding='SAME', activation=tf.keras.activations.elu, training=True):
        """Define spectral normalization conv for discriminator.
        Args:
            x: Input.
            cnum: Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            Rate: Rate for or dilated conv.
            name: Name of layers.
            padding: Default to SYMMETRIC.
            activation: Activation function after convolution.
            training: If current graph is for training or inference, used for bn.
        Returns:
            tf.Tensor: output
        """
        assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
        super(SNConv, self).__init__()

        self.cnum = cnum
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.layer_name = name
        self.padding = padding
        self.activation = activation
        self.training = training
        self.w = None

    def build(self, input_shape):
        fan_in = self.ksize * self.ksize * input_shape.as_list()[-1]
        stddev = np.sqrt(2. / (fan_in))
        self.w = tf.random.normal(shape=[self.ksize, self.ksize, input_shape[-1], self.cnum], stddev=stddev)

    def call(self, x):

        padding = self.padding
        if self.padding == 'SYMMETRIC' or self.padding == 'REFELECT':
            p = int(self.rate * (self.ksize - 1) / 2)
            x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=self.padding)
            padding = 'VALID'

        # initializer for w used for spectral normalization
        # if not self.w:
        #   fan_in = self.ksize * self.ksize * x.get_shape().as_list()[-1]
        #   stddev = np.sqrt(2. / (fan_in))
        #   self.w = tf.random.normal(shape=[self.ksize, self.ksize, x.get_shape()[-1], self.cnum], stddev=stddev)

        x = tf.nn.conv2d(x, spectral_normed_weight(self.w, update_collection=tf.compat.v1.GraphKeys.UPDATE_OPS,
                                                   name=self.layer_name + "_sn_w"),
                         strides=[1, self.stride, self.stride, 1],
                         dilations=[1, self.rate, self.rate, 1],
                         padding=padding,
                         name=self.layer_name)
        x = self.activation(x)
        return x


class SNDense(tf.keras.layers.Layer):

    def __init__(self, out_dim, name='dense', activation=tf.keras.activations.sigmoid, training=True):
        """Define spectral normalization conv for discriminator.
        Args:
            out_dim: Output dimension.
            name: Name of layers.
            activation: Activation function after convolution.
            training: If current graph is for training or inference, used for bn.
        Returns:
            tf.Tensor: output
        """
        super(SNDense, self).__init__()

        self.out_dim = out_dim
        self.layer_name = name
        self.activation = activation
        self.training = training
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = tf.random.normal(shape=[input_shape[-1], self.out_dim])
        self.b = tf.random.normal(shape=[self.out_dim, ])

    def call(self, x):

        # initializer for w used for spectral normalization
        # if not self.w:
        #    self.w = tf.random.normal(shape=[x.get_shape()[-1], self.out_dim])
        # if not self.b:
        #    self.b = tf.random.normal(shape=[self.out_dim, ])

        x = tf.matmul(x, spectral_normed_weight(self.w, update_collection=tf.compat.v1.GraphKeys.UPDATE_OPS,
                                                name=self.layer_name + "_sn_w")) + self.b
        self.activation(x)
        return x
