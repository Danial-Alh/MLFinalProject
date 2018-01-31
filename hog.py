import os
import struct
from array import array as pyarray

import cv2
import numpy as np
from numpy import array, int8, uint8, zeros
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from models import GMM, purity_score


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

    images = zeros((N, rows, cols), dtype=int8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        image = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

        images[i] = image
        # print(image)
        # print(images[i])
    temp = np.array(list(zip(images, labels)))
    np.random.shuffle(temp)
    images, labels = zip(*temp)
    return np.array(images), np.array(labels)


if __name__ == '__main__':
    train_imgs, train_labels = load([i for i in range(10)])
    train_imgs, train_labels = train_imgs[:int(train_imgs.shape[0] * .4)], train_labels[
                                                                            :int(train_imgs.shape[0] * .4)]
    print("loaded!")
    train_imgs = train_imgs + 128
    train_imgs = np.array(train_imgs, dtype=uint8)
    train_imgs = np.array(train_imgs)
    im = train_imgs[0]
    winSize = (28, 28)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (2, 2)
    nbins = 7
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 7
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    hogs = []
    for img in train_imgs:
        temp = hog.compute(img)
        hogs.append(temp)
    hogs = np.array(hogs)
    # gmm = mixture.GaussianMixture(n_components=10, verbose=True, max_iter=120)
    # gmm = GMM(n_components=10, max_iter=120)
    hogs = hogs.reshape([hogs.shape[0], hogs.shape[1]])
    # counter = 1
    # while True:
    #     pca = PCA(n_components=counter)
    #     w = pca.fit_transform(hogs)
    #     gmm.fit(w)
    #     l = gmm.predict(w)
    #     train_labels = train_labels.reshape(60000)
    #     print(counter)
    #     print(purity_score(l, train_labels))
    #     print(adjusted_rand_score(l, train_labels))
    #     counter = counter + 1
    #     print(counter)
    #     if counter == 100:
    #         break

    kf = KFold(n_splits=3, shuffle=True)
    puritites = []
    pars = [10, 30, 40, 50, 60]
    for par in pars:
        print(par)
        r = []
        pca = PCA(n_components=par)
        w = pca.fit_transform(hogs)
        for train_index, test_index in kf.split(w):
            print("TRAIN:", train_index, "TEST:", test_index)
            hog_temp_train, hog_temp_test = w[train_index], w[test_index]
            y_temp_train, y_temp_test = train_labels[train_index], train_labels[test_index]
            gmm = GMM(n_components=10, max_iter=120)
            gmm.fit(hog_temp_train)
            predicted = gmm.predict(hog_temp_test)
            acc = purity_score(predicted, y_temp_test)
            r.append(acc)
            print("total purity: {}".format(acc))
        r = np.array(r)
        puritites.append(r.mean())
        print("after")
    acc_max_arg = np.argmax(puritites)
    print("best {}: {}".format(pars[acc_max_arg], puritites[acc_max_arg]))
    for a, c in zip(puritites, pars):
        print("{}: {}".format(c, a))
