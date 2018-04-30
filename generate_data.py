import numpy as np


def generate_data_square(row_per_class):
    # rows from the first label
    labels_0_1 = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    labels_0_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    # rows from the second label
    labels_1_1 = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    labels_1_1 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([labels_0_1, labels_0_2, labels_1_1, labels_1_1])
    labels = np.concatenate(
        (np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    labels = labels.reshape(-1, 1)

    return features, labels


def generate_data_circle(row_per_class):
    x = np.concatenate(
        (np.random.rand(row_per_class), np.random.rand(row_per_class)*3+2))
    theta = np.random.rand(2*row_per_class)*2*np.pi

    features = np.column_stack((x*np.cos(theta), x*np.sin(theta)))

    labels = np.concatenate(
        (np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    labels = labels.reshape(-1, 1)

    return features, labels


def generate_data_moon(row_per_class):

    g1 = np.random.uniform(- 0.1, 1.1*np.pi, row_per_class)
    g2 = np.random.uniform((0.9*np.pi), (2*np.pi+0.1), row_per_class)

    r1 = np.random.uniform(1, 2, row_per_class)
    r2 = np.random.uniform(1, 2, row_per_class)

    x = np.concatenate((r1*np.cos(g1), 1.5+r2*np.cos(g2)))
    y = np.concatenate((r1*np.sin(g1), 0.3+r2*np.sin(g2)))

    features = np.column_stack((x, y))

    labels = np.concatenate(
        (np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    labels = labels.reshape(-1, 1)

    return features, labels


def generate_data_diffsize(row_per_class):

    features = np.concatenate(
        (np.random.randn(row_per_class, 2), np.random.randn(row_per_class, 2)*0.2+2))

    labels = np.concatenate(
        (np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    labels = labels.reshape(-1, 1)

    return features, labels
