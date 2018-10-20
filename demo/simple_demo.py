from __init__ import gan_impl
import tensorflow as tf
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import time

scope = "gan"
training_iterations = 1000

GAN = gan_impl.ClassicGan([None, 1], [None, 1], name_scope=scope)

validation_batch = np.random.normal(0, 1, [2000, 1])
label_batch = np.random.exponential(1, [2000, 1])

# validation_batch = np.random.exponential(1, [2000, 1])
# label_batch = np.random.normal(3, 1, [2000, 1])
# validation_batch = np.random.normal(0, 1, [2000, 1])

target = plt.axes()
current = plt.axes()
transfer = plt.axes()

target.set_ylim(0, 2)
current.set_ylim(0, 2)
transfer.set_ylim(0, 2)

seaborn.distplot(label_batch.reshape(-1), kde=True, ax=target, color="g")
seaborn.distplot(validation_batch.reshape(-1), kde=True, ax=current, color="r")

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    seaborn.distplot(
        GAN.generate(validation_batch).reshape(-1),
        kde=True,
        ax=transfer,
        color="y")

    plt.pause(0.001)

    for i in range(training_iterations):
        transfer.clear()
        transfer.set_ylim(0, 2)
        transfer.set_xlim(-4, 4)
        """ SGD on all batches """
        total_dscore, total_gscore = GAN.update(label_batch, validation_batch)

        print("Iteration: {}\nDiscriminator Score: {}\nGenerator Score: {}\n".
              format(i, total_dscore, total_gscore))

        seaborn.distplot(
            label_batch.reshape(-1), kde=True, ax=target, color="g")
        seaborn.distplot(
            validation_batch.reshape(-1), kde=True, ax=current, color="r")
        seaborn.distplot(
            GAN.generate(validation_batch).reshape(-1),
            kde=True,
            ax=transfer,
            color="y")

        plt.pause(0.001)

plt.show()
