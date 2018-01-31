import cv2
import numpy as np
from numpy import uint8
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score

from feature_extractor import load
from models import GMM, purity_score

if __name__ == '__main__':
    train_imgs, train_labels = load([i for i in range(10)])
    train_imgs, train_labels = train_imgs[:int(train_imgs.shape[0] * .02)], train_labels[
                                                                            :int(train_imgs.shape[0] * .02)]
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
    gmm = GMM(n_components=10, max_iter=120)
    hogs = hogs.reshape([60000, hogs.shape[1]])
    counter = 1
    while True:
        pca = PCA(n_components=counter)
        w = pca.fit_transform(hogs)
        gmm.fit(w)
        l = gmm.predict(w)
        train_labels = train_labels.reshape(60000)
        print(counter)
        print(purity_score(l, train_labels))
        print(adjusted_rand_score(l, train_labels))
        counter = counter + 1
        print(counter)
        if counter == 100:
            break
