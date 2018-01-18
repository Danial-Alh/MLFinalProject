import cv2
import os
import struct
from array import array as pyarray
from random import randint
import numpy as np
from cv2.cv2 import imshow
from numpy import array, int8, uint8, zeros


# load MNIST dataset

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

    images = zeros((N, (rows+20)*4, (cols+20)*4), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        image = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
        image= np.pad(image,[10,10],mode='constant')
        images[i]  = image.repeat(4, axis = 0).repeat(4, axis = 1)
        # print(image)
        # print(images[i])
    temp = np.array(list(zip(images, labels)))
    np.random.shuffle(temp)
    images, labels = zip(*temp)
    return np.array(images), np.array(labels)


train_imgs, train_labels = load([i for i in range(10)], "training")
is_cv3 = cv2.__version__.startswith("3.")
if is_cv3:
    surf = cv2.xfeatures2d.SURF_create(400)
else:
    surf = cv2.xfeatures2d_SURF(400)
kds = []
dss = []
for img in train_imgs:
    kd,ds = surf.detectAndCompute(img,None)
    kds.append(kd)
    dss.append(ds)
    # print(kd)
    # print(ds)