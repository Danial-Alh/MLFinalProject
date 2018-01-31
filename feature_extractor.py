import os
import struct
from array import array as pyarray

import cv2
import numpy as np
from numpy import array, int8, uint8, zeros

portion = 1.0 / 1


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
