import os
import struct
from array import array as pyarray

import cv2
import numpy as np
from numpy import array, int8, uint8, zeros
from sklearn.metrics import accuracy_score

from models import WindowBasedEnsembleClassifier

portion = 1.0 / 20


def get_feature_from_kd_ds(kd, ds):
    # f = [kd.pt[0], kd.pt[1], kd.angle, kd.size, kd.response]
    f = [kd.angle, kd.size, kd.response]
    # f = [kd.size, kd.response]
    f.extend(ds)
    # return ds
    return f


def load(digits, dataset="training", path="."):
    """
    Loads MNIST files into 3D numpy arrays
    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, (rows + 20) * 4, (cols + 20) * 4), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        image = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
        image = np.pad(image, [10, 10], mode='constant')
        images[i] = image.repeat(4, axis=0).repeat(4, axis=1)
        # print(image)
        # print(images[i])
    temp = np.array(list(zip(images, labels)))
    np.random.shuffle(temp)
    images, labels = zip(*temp)
    return np.array(images), np.array(labels)


def extract_features(dataset='training'):
    if os.path.exists(dataset + '_kds.npy'):
        print(dataset + ' objs found!')
        temp_kds, dss, train_labels = np.load(dataset + '_kds.npy'), np.load(dataset + '_dss.npy'), np.load(
            dataset + '_labels.npy')
        kds = np.array(
            [[cv2.KeyPoint(x=kp[0][0], y=kp[0][1], _size=kp[1], _angle=kp[2], _response=kp[3], _octave=kp[4],
                           _class_id=kp[5]) for kp in kd] for kd in temp_kds])
        print(dataset + " size: {}".format(kds.shape[0]))
        return kds, dss, train_labels
    print(dataset + ' objs not found!')
    train_imgs, train_labels = load([i for i in range(10)], dataset)
    hessian_threshold, nOctaves, nOctaveLayers, extended, upright = (400, 4, 6, False, True)
    is_cv3 = cv2.__version__.startswith("3.")
    if is_cv3:
        surf = cv2.xfeatures2d.SURF_create(hessian_threshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                           extended=extended, upright=upright)
    else:
        surf = cv2.xfeatures2d_SURF(hessian_threshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                    extended=extended, upright=upright)
    kds = []
    dss = []
    train_imgs = train_imgs[:int(train_imgs.shape[0] * portion)]
    train_labels = train_labels[:int(train_labels.shape[0] * portion)]
    for i, img in enumerate(train_imgs):
        if i % 1000 == 0:
            print(i)
        kd, ds = surf.detectAndCompute(img, None)
        kds.append(kd)
        dss.append(ds)
    kds = np.array(kds)
    dss = np.array(dss)
    temp_kds = np.array(
        [[(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kd] for kd in kds])
    np.save(dataset + '_kds', temp_kds)
    np.save(dataset + '_dss', dss)
    np.save(dataset + '_labels', train_labels)
    print(dataset + " size: {}".format(kds.shape[0]))
    return kds, dss, train_labels


if __name__ == '__main__':
    train_kds, train_dss, train_labels = extract_features()
    test_kds, test_dss, test_labels = extract_features('testing')
    # print(np.bincount(train_labels.flatten()))

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
    # k = 50
    # new_x = np.array(new_x)
    # new_y = np.array(new_y)
    #
    # model = RandomForestClassifier()
    # model = KNeighborsClassifier(k, n_jobs=2)
    # model = KNN()
    # model = svm.LinearSVC(verbose=True, max_iter=10000)
    # model.fit(new_x, new_y)

    model = WindowBasedEnsembleClassifier(False, 8, 8, 192, 192, 1.0 / 7, 1.0 / 7)
    model.fit(train_kds, train_dss, train_labels)
    # AdaBoostClassifier()
    # predicted = []
    # for i in range(test_kds.shape[0]):
    #     if i % 1000 == 0:
    #         print(i)
    #     new_x = []
    #     for j in range(len(test_kds[i])):
    #         # new_x.append(test_dss[i][j])
    #         f = [t.pt[0], t.pt[1], t.angle, t.size, t.response]
    #         f.extend(test_dss[i][j])
    #         new_x.append(f)
    #     temp_predicted = model.predict(np.array(new_x))
    #     label = np.argmax(np.bincount(temp_predicted))
    #     predicted.append(label)
    predicted = model.predict(test_kds, test_dss)
    acc = accuracy_score(test_labels, predicted)
    predicted = np.array(predicted)
    print("total accuracy: {}".format(acc))
    for i in range(10):
        t_ids = [m for m, t in enumerate(test_labels) if t == i]
        t_acc = accuracy_score(test_labels[t_ids], predicted[t_ids])
        print("number {} accuracy: {}".format(i, t_acc))
