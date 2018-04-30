import matplotlib.pyplot as plt
import tensorflow as tf
# import numpy as np
from sklearn.model_selection import train_test_split

from generate_data import *


if __name__ == '__main__':

    # Number of row per class
    row_per_class = 200
    # Number of hidden neur
    nb_hidden_neur = 3

    # Generate 2D dataset
    features, labels = generate_data_circle(row_per_class)
    # Plot points
    plt.scatter(features[:, 0], features[:, 1], c=labels[:, 0])
    plt.show()

    tf_features = tf.placeholder(tf.float32, shape=[None, 2])
    tf_labels = tf.placeholder(tf.float32, shape=[None, 1])

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.25, random_state=123)

    # First layer
    w1 = tf.Variable(tf.random_normal([2, nb_hidden_neur]))
    b1 = tf.Variable(tf.zeros([nb_hidden_neur]))

    # Matrix Product + Sigmoid output
    z1 = tf.matmul(tf_features, w1) + b1
    a1 = tf.nn.sigmoid(z1)

    # Output neuron
    w2 = tf.Variable(tf.random_normal([nb_hidden_neur, 1]))
    b2 = tf.Variable(tf.zeros([1]))
    # Operations
    z2 = tf.matmul(a1, w2) + b2
    py = tf.nn.sigmoid(z2)

    MSE = tf.reduce_mean(tf.square(py - tf_labels))

    correct_prediction = tf.equal(tf.round(py), tf_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(MSE)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training
    epochs = 1000
    for i in range(epochs):
        sess.run(train, feed_dict={
            tf_features: features_train,
            tf_labels: labels_train
        })

# Test
print("accuracy =", sess.run(accuracy, feed_dict={
    tf_features: features_test,
    tf_labels: labels_test
}))
