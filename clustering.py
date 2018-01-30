import numpy as np

from classification import extract_features
from models import WindowBasedEnsembleClustering


def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes

    :param clusters: the cluster assignments array
    :type clusters: numpy.array

    :param classes: the ground truth classes
    :type classes: numpy.array

    :returns: the purity score
    :rtype: float
    """

    A = np.c_[(clusters, classes)]

    n_accurate = 0.

    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]


if __name__ == '__main__':
    train_kds, train_dss, train_labels = extract_features()
    # new_x = []
    # new_y = []
    # for i in range(train_kds.shape[0]):
    #     for j in range(len(train_kds[i])):
    #         t = train_kds[i][j]
    #         # new_x.append(train_dss[i][j])
    #         f = [t.pt[0], t.pt[1], t.angle, t.size, t.response]
    #         f.extend(train_dss[i][j])
    #         new_x.append(f)
    #         new_y.append(train_labels[i])
    # new_x = np.array(new_x)
    # new_y = np.array(new_y)
    #
    #
    # model = KM(10, verbose=True)
    # model.fit(new_x)
    # predicted = []
    # for i in range(train_kds.shape[0]):
    #     if i % 1000 == 0:
    #         print(i)
    #     new_x = []
    #     for j in range(len(train_kds[i])):
    #         # new_x.append(test_dss[i][j])
    #         f = [t.pt[0], t.pt[1], t.angle, t.size, t.response]
    #         f.extend(train_dss[i][j])
    #         new_x.append(f)
    #     temp_predicted = model.predict(np.array(new_x))
    #     label = np.argmax(np.bincount(temp_predicted))
    #     predicted.append(label)
    # purity_score(predicted, train_labels.flatten())
    model = WindowBasedEnsembleClustering(10, True, 10, 6, 6, 192, 192, 1 / 5.0, 1 / 5.0)
    model.fit(train_kds, train_dss, train_labels)
    print(purity_score(model.predict(train_kds, train_dss), train_labels.flatten()))
