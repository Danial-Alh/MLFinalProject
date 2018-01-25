import os
import struct
from array import array as pyarray

import cv2
import numpy as np
from numpy import array, int8, uint8, zeros
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors.kd_tree import KDTree

portion = 1.0 / 8


class KNN:
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.kd_tree = None
        self.p = None

    def fit(self, X, Y, k, p):
        self.train_X = X
        self.train_Y = Y
        self.p = p
        self.k = k
        self.kd_tree = KDTree(self.train_X, metric='minkowski', p=p)

    def predict(self, X):
        # print("test started!")
        # predicts = Parallel(n_jobs=4, verbose=10)(delayed(self.inner_predict)(j, d) for j, d in enumerate(X))
        predicts = [self.inner_predict(j, d) for j, d in enumerate(X)]
        return np.array(predicts)

    def inner_predict(self, id, x):
        # if id % 10 == 0:
        #     print(id)
        # t1 = time.time()
        # costs = np.array([temp_func(i, t_x, x, p) for i, t_x in enumerate(self.train_X)])
        # top_k = []
        # for _ in range(k):
        #     min_arg = int(costs[:, 1].argmin())
        #     top_k.append(int(costs[min_arg][0]))
        #     costs = np.delete(costs, min_arg)
        top_k = self.kd_tree.query([x], k=self.k)[1]
        # t2 = time.time()
        # print("elapased: {}".format(t2 - t1))
        candidates = np.array(self.train_Y[top_k])
        candidates = candidates.flatten()
        predict = np.argmax(np.bincount(candidates))

        return predict


class WindowBasedEnsembleLearner:
    def __init__(self, n_windows_in_row, n_windows_in_col, img_width, img_height, window_width_to_img_width,
                 window_height_to_img_height):
        self.n_windows_in_row = n_windows_in_row
        self.n_windows_in_col = n_windows_in_col
        self.classifiers = [[None for _ in range(n_windows_in_col)] for _ in range(n_windows_in_row)]
        self.window_width = window_width_to_img_width * img_width
        self.window_height = window_height_to_img_height * img_height
        self.window_centers = [
            [(x, y) for x in np.linspace(self.window_width / 2, img_width - self.window_width / 2, n_windows_in_col)]
            for y in np.linspace(self.window_height / 2, img_height - self.window_height / 2, n_windows_in_row)]

    def is_point_in_window(self, pt, w_i, w_j):
        if self.window_centers[w_i][w_j][0] - self.window_width / 2 \
                <= pt[0] <= self.window_centers[w_i][w_j][0] + self.window_width / 2 \
                and self.window_centers[w_i][w_j][1] - self.window_height / 2 \
                <= pt[1] <= self.window_centers[w_i][w_j][1] + self.window_height / 2:
            return True
        return False

    def fit(self, kds, dss, Y):
        for i in range(self.n_windows_in_row):
            for j in range(self.n_windows_in_col):
                window_id = (i, j)
                print("create window {} dataset!".format(window_id))
                new_x = []
                new_y = []
                for img_id in range(kds.shape[0]):
                    for m, kd in enumerate(kds[img_id]):
                        if self.is_point_in_window(kd.pt, i, j):
                            new_x.append(dss[img_id][m])
                            new_y.append(Y[img_id])
                if len(new_x) == 0:
                    print("window {}; not data found!".format(window_id))
                    continue
                new_x = np.array(new_x)
                new_y = np.array(new_y)
                # if new_y.max() == new_y.min():
                #     print("window {}; just one class!".format(window_id))
                #     continue
                print("trainig window {} classifier with {} data!".format(window_id, new_x.shape[0]))
                window_classifier = RandomForestClassifier()
                # window_classifier = KNeighborsClassifier(np.min([50, new_x.shape[0]]))
                # window_classifier = svm.SVC(C=np.power(10.0, -6), kernel='linear')
                window_classifier.fit(new_x, new_y)
                print("window {} classifier trained!".format(window_id))
                self.classifiers[i][j] = window_classifier

    def predict(self, kds, dss):
        predicted = []
        for img_id in range(kds.shape[0]):
            votes = []
            for i in range(self.n_windows_in_row):
                for j in range(self.n_windows_in_col):
                    if self.classifiers[i][j] is None:
                        continue
                    new_x = []
                    for m, kd in enumerate(kds[img_id]):
                        if self.is_point_in_window(kd.pt, i, j):
                            new_x.append(dss[img_id][m])
                    if len(new_x) > 0:
                        votes.extend(self.classifiers[i][j].predict(np.array(new_x)))
            vote = np.argmax(np.bincount(votes))
            predicted.append(vote)
        return predicted


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


train_kds, train_dss, train_labels = extract_features()
test_kds, test_dss, test_labels = extract_features('testing')
print(np.bincount(train_labels.flatten()))

# new_x = []
# new_y = []
# for i in range(train_kds.shape[0]):
#     for j in range(len(train_kds[i])):
#         t = train_kds[i][j]
#         new_x.append(train_dss[i][j])
#         new_y.append(train_labels[i])
# k = 50
# new_x = np.array(new_x)
# new_y = np.array(new_y)

# model = RandomForestClassifier()
# model = KNeighborsClassifier(k, n_jobs=2)
# model = KNN()
# model = svm.LinearSVC(verbose=True, max_iter=10000)
model = WindowBasedEnsembleLearner(8, 8, 192, 192, 1. / 6, 1. / 6)
model.fit(train_kds, train_dss, train_labels)

# predicted = []
# for i in range(test_kds.shape[0]):
#     if i % 1000 == 0:
#         print(i)
#     new_x = []
#     for j in range(len(test_kds[i])):
#         new_x.append(test_dss[i][j])
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
