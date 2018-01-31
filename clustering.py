import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score

from feature_extractor import load
from models import GMM, purity_score

if __name__ == '__main__':
    # train_kds, train_dss, train_labels = classification.extract_features()
    # data = []
    # for i in range(train_dss.shape[0]):
    #     for j in range(len(train_dss[i])):
    #         data.append(train_dss[i][j])
    # data = np.array(data)

    # kmeans = KMeans()
    # kmeans.fit(data,10)
    # label_kmeans = []
    # counter =0
    # for i in range(train_kds.shape[0]):
    #     label_temp = []
    #     for j in range(len(train_kds[i])):
    #         label_temp.append(kmeans.classes[counter])
    #         ++counter;
    #         label_kmeans.append(np.argmax(np.bincount(label_temp)))

    train_imgs, train_labels = load([i for i in range(10)])
    train_imgs = train_imgs.reshape((train_imgs.shape[0], train_imgs.shape[1] * train_imgs.shape[2]))
    # train_imgs = train_imgs +128
    # model = NMF(n_components=20, init='random', random_state=0, verbose=True)
    # W = model.fit_transform(X=train_imgs)
    # H = model.components_
    #
    # # test_imgs, test_labels = load([i for i in range(10)],'testing')
    # W = np.array(W)
    #
    # print(W.shape)

    train_labels = train_labels.reshape(60000)

    label_gmm = []
    # gmm = mixture.GaussianMixture(n_components=10, verbose=True, max_iter=120)
    gmm = GMM(n_components=10, max_iter=120)
    counter = 1
    while True:
        pca = PCA(n_components=counter)
        W = pca.fit_transform(train_imgs)
        gmm.fit(W)
        l = gmm.predict(W)

        # for i in range(train_dss.shape[0]):
        #     label_temp = []
        #     for j in range(len(train_dss[i])):
        #         label_temp.append(l[counter])
        #         counter += 1
        #     temp = np.argmax(np.bincount(label_temp))
        #     label_gmm.append(temp)
        # label_gmm = np.array(label_gmm)
        # # label_gmm = label_gmm.reshape(7500)
        # # train_labels = np.array(train_labels)
        # train_labels = train_labels.flatten()
        # print(label_gmm.shape)
        # print(train_labels.shape)
        l = np.array(l)
        l = l.reshape(60000)
        print(counter)
        print(purity_score(l, train_labels))
        print(adjusted_rand_score(l, train_labels))
        counter = counter + 1
