import numpy as np
from scipy import stats


class KMeans:
    def __init__(self):
        self.classes = None
        self.centroids = None

    def fit(self, X: np.ndarray, k):
        self.centroids = [[0.0 for _ in range(X.shape[1])] for _ in range(k)]
        for col_id in range(X.shape[1]):
            col_max, col_min = np.max(X[:, col_id]), np.min(X[:, col_id])
            for centroid in self.centroids:
                centroid[col_id] = np.random.random() * (col_max - col_min) + col_min
        # print(self.centroids)
        it = 0
        should_continue = True
        while should_continue:
            if it % 100 == 0:
                print(it)
            distances = np.array(
                [[np.linalg.norm(x - self.centroids[cluster_index]) for x in X] for cluster_index in range(k)])
            classes = [[] for _ in range(k)]
            for i, x in enumerate(X):
                target_cluster_id = np.argmin([distances[cluster_index, i] for cluster_index in range(k)])
                classes[target_cluster_id].append(i)
            self.classes = np.array(classes)
            delta = 0.0
            for i in range(k):
                prev = self.centroids[i]
                self.centroids[i] = np.mean(X[classes[i]], axis=0)
                delta = np.max([delta, np.linalg.norm(self.centroids[i] - prev)])
            if 0 <= delta < threshold:
                should_continue = False
            it += 1
        print(self.centroids)


class GMM:
    def __init__(self):
        self.means = None
        self.covars = None
        self.pies = None
        self.gammas = None
        self.classes = None

    def fit(self, X, k):
        self.means = [[0.0 for _ in range(X.shape[1])] for _ in range(k)]
        for col_id in range(X.shape[1]):
            col_max, col_min = np.max(X[:, col_id]), np.min(X[:, col_id])
            for mean in self.means:
                mean[col_id] = np.random.random() * (col_max - col_min) + col_min
        self.covars = [np.ones((X.shape[1], X.shape[1])) for _ in range(k)]
        self.gammas = np.zeros((k, X.shape[0]))
        self.pies = np.random.random(k)
        self.pies = self.pies / self.pies.sum()
        print(self.means)
        log_likelihood = -np.inf
        it = 0
        while 1:
            if it % 30 == 0:
                print(it)
                print(log_likelihood)
            # expectation
            rv = [stats.multivariate_normal(self.means[j], self.covars[j], True) for j in range(k)]
            probs = np.array([[rv[j].pdf(x) for x in X] for j in range(k)])
            denominator = probs.T.dot(self.pies)
            prev_log_likelihood = log_likelihood
            log_likelihood = np.log(denominator).sum()
            log_likelihood_changes = log_likelihood - prev_log_likelihood
            if 0 <= log_likelihood_changes < threshold:
                break
            for i in range(X.shape[0]):
                for j in range(k):
                    self.gammas[j, i] = self.pies[j] * probs[j, i] / denominator[i]

            # maximization
            N = self.gammas.sum(axis=1)
            self.means = self.gammas.dot(X) / np.repeat(N, 2, axis=0).reshape(k, X.shape[1])
            for j in range(k):
                self.covars[j] = np.zeros((X.shape[1], X.shape[1]))
                for i in range(X.shape[0]):
                    xm = (X[i] - self.means[j]).reshape((X.shape[1], 1))
                    self.covars[j] += self.gammas[j, i] * xm.dot(xm.T)
                self.covars[j] /= N[j]
            self.pies = N / X.shape[0]
            it += 1
        print(self.means)
        print(it)

        self.classes = [[] for _ in range(k)]
        rv = [stats.multivariate_normal(self.means[j], self.covars[j], True) for j in range(k)]
        for i, x in enumerate(X):
            cluster_id = np.argmax([self.pies[c_id] * rv[c_id].pdf(x) for c_id in range(k)])
            self.classes[cluster_id].append(i)


def purity(predicted_classes, labels, k):
    result = 0.0
    for predicted_c_id in range(len(predicted_classes)):
        ni = np.max([np.sum([1 for i in predicted_classes[predicted_c_id] if j == labels[i]]) for j in range(k)])
        result += ni / labels.shape[0]
    return result


def rand_index(predicted_classes, labels, k):
    from scipy.misc import comb
    tpfp = 0
    fnfn = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for predicted_class in predicted_classes:
        ni = [np.sum([1 for i in predicted_class if j == labels[i]]) for j in range(k)]
        tpfp += comb(len(predicted_class), 2)
        tp += np.sum([comb(n, 2) for n in ni])
    fp = tpfp - tp
    for c1 in predicted_classes:
        for ux in c1:
            for c2 in predicted_classes:
                for vx in c2:
                    if ux == vx:
                        continue
                    if c1 != c2 and labels[ux] == labels[vx]:
                        fn += 1
    tnfn = comb(labels.shape[0], 2) - tpfp
    tn = tnfn - fn
    return (tp + tn) / (tn + tp + fp + fn)
