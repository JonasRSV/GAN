from overrides import overrides
import gan
import tensorflow as tf


class ClassicGan(gan.Gan):
    """ Based on https://arxiv.org/pdf/1406.2661.pdf """

    def __init__(self,
                 input_shape,
                 output_shape,
                 discriminator_lr=0.1,
                 generator_lr=0.01,
                 name_scope="gan"):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.discriminator_lr = discriminator_lr
        self.generator_lr = generator_lr
        self.name_scope = name_scope

        self.false_distribution = tf.placeholder(
            dtype=tf.float32, shape=input_shape)
        self.generator = self.__generator(self.false_distribution)

        self.true_distribution = tf.placeholder(
            dtype=tf.float32, shape=input_shape)
        self.true_guess, self.false_guess = self.__discriminator(
            self.true_distribution, self.generator)

        self.discriminator_score = tf.reduce_mean(
            tf.math.log(self.true_guess) + tf.math.log(1 - self.false_guess))
        self.generator_score = tf.reduce_mean(1 - tf.math.log(self.false_guess))

        discriminator_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="{}/discriminator".format(self.name_scope))
        generator_variabels = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="{}/generator".format(self.name_scope))

        discriminator_gradients = tf.gradients(-self.discriminator_score,
                                               discriminator_variables)
        generator_gradients = tf.gradients(self.generator_score,
                                           generator_variabels)

        self.discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.discriminator_lr).apply_gradients(
                zip(discriminator_gradients, discriminator_variables))
        self.generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.generator_lr).apply_gradients(
                zip(generator_gradients, generator_variabels))

    def __generator(self, input_tensor):
        with tf.variable_scope("{}/generator".format(self.name_scope)):
            x = tf.layers.dense(input_tensor, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, self.input_shape[-1], activation=None)

        return x

    def __discriminator(self, true_input_tensor, false_input_tensor):
        activation = tf.nn.sigmoid
        h1_units = 64

        with tf.variable_scope(
                "{}/discriminator".format(self.name_scope),
                reuse=tf.AUTO_REUSE):
            true = tf.layers.dense(
                true_input_tensor, h1_units, activation=tf.nn.tanh)
            true = tf.layers.dense(
                true, self.output_shape[-1], activation=activation)

        with tf.variable_scope(
                "{}/discriminator".format(self.name_scope),
                reuse=tf.AUTO_REUSE):
            false = tf.layers.dense(
                false_input_tensor, h1_units, activation=tf.nn.tanh)
            false = tf.layers.dense(
                false, self.output_shape[-1], activation=activation)

        return true, false

    @overrides
    def generate(self, samples):
        return tf.get_default_session().run(
            self.generator, feed_dict={
                self.false_distribution: samples
            })

    @overrides
    def discriminate(self, samples):
        return tf.get_default_session().run(
            self.true_guess, feed_dict={
                self.true_distribution: samples
            })

    @overrides
    def update(self, true_samples, false_samples):
        dscore, gscore, _, _ = tf.get_default_session().run(
            (self.discriminator_score, self.generator_score,
             self.discriminator_optimizer, self.generator_optimizer),
            feed_dict={
                self.false_distribution: false_samples,
                self.true_distribution: true_samples
            })
        return dscore, gscore
