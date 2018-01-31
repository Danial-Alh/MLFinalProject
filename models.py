import numpy as np
from numpy.linalg.linalg import eig
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.mixture import GaussianMixture

from feature_extractor import get_feature_from_kd_ds, extract_features


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


def get_value_frequency(data):
    classes = np.unique(data)
    class_frequency = [0 for _ in classes]
    for d in data:
        for i, c in enumerate(classes):
            if d == c:
                class_frequency[i] += 1
    return np.array(classes), np.array(class_frequency, dtype=float)


def entropy(Y, class_frequency=None):
    if class_frequency is None:
        class_frequency = get_value_frequency(Y)[1]
    else:
        class_frequency = class_frequency[np.nonzero(class_frequency)]
    p = class_frequency / class_frequency.sum()
    e = - (p * np.log2(p)).sum()
    return e


def discrete_attribute_gain(X, Y, att_index):
    classes, value_sizes = get_value_frequency(X[:, att_index])
    entropies = np.array([entropy(np.array([Y[i] for i, x in enumerate(X) if x[att_index] == c])) for c in classes])
    return entropy(Y) - (value_sizes / X.shape[0] * entropies).sum()


def continues_attribute_gain(X, Y, att_index):
    sorted_indexes = np.argsort(X[:, att_index])
    thresholds = [X[:, att_index][sorted_indexes[0]]]
    temp_left_Y, temp_right_Y = np.array([], dtype=int), np.array(list(sorted_indexes), dtype=int)
    right_classes_frequencies = get_value_frequency(Y)[1]
    left_classes_frequencies = np.zeros(right_classes_frequencies.shape[0])
    prev_data_label = Y[sorted_indexes[0]][0]
    Y_entropy = entropy(Y, right_classes_frequencies)
    gains = np.array([0.0])
    print("before!")
    for i in range(sorted_indexes.shape[0]):
        data_id = int(sorted_indexes[i])
        if i != 0 and prev_data_label != Y[data_id][0]:
            thresholds.append((X[sorted_indexes[i], att_index] + X[sorted_indexes[i - 1]]) / 2.0)
            gains = np.append(gains,
                              Y_entropy - len(temp_left_Y) / X.shape[0] * entropy(Y[temp_left_Y],
                                                                                  left_classes_frequencies) - len(
                                  temp_right_Y) / X.shape[
                                  0] * entropy(Y[temp_right_Y], right_classes_frequencies))
            prev_data_label = Y[data_id][0]
        temp_left_Y = np.append(temp_left_Y, data_id)
        temp_right_Y = temp_right_Y[1:]
        left_classes_frequencies[Y[data_id]] += 1
        right_classes_frequencies[Y[data_id]] -= 1
    print("reached {}!".format(len(gains)))
    max_gain_arg = np.argmax(gains)
    return gains[max_gain_arg], thresholds[max_gain_arg]


class DecisionTree:
    class Node:
        def __init__(self, depth, classes, class_value_frequency,
                     is_leaf=False, att_index=None, children=None, vote=None):
            self.depth = depth
            self.classes = classes
            self.class_frequency = class_value_frequency
            self.att_index = att_index
            self.children = children
            self.is_leaf = is_leaf
            if vote is None:
                self.vote = classes[np.argmax(class_value_frequency)]
            else:
                self.vote = vote

        def __str__(self):
            result = "\n"
            result += "depth: {}, is leaf: {}, classes: {}, class freq: {}\n". \
                format(self.depth + 1, self.is_leaf, self.classes, self.class_frequency)
            result += "vote: {}\n".format(self.vote)
            if not self.is_leaf:
                result += "att id: {}\n".format(self.att_index)
                for child in self.children:
                    result += "** att value: {} **\n".format(child[0])
                    child_str = str(child[1]) + "\n"
                    child_str = child_str.replace("\n", "\n\t")[:-1]
                    result += child_str

            # indentation = ""
            # for _ in range(self.depth):
            #     indentation += "\t"
            # result = result.replace("\n", "\n" + indentation)
            return result

    def __init__(self, min_samples_leaf=1):
        self.X = None
        self.Y = None
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.__build_tree()

    def __build_tree(self):
        all_indices = [i for i in range(self.X.shape[0])]
        self.root = self.__build_node(None, all_indices, 0)

    def __build_node(self, parent_vote, indices, depth):
        if len(indices) == 0:
            return self.Node(depth, [], [], True, vote=parent_vote)
        classes, class_frequency = get_value_frequency(self.Y[indices])
        if len(indices) <= self.min_samples_leaf:
            return self.Node(depth, classes, class_frequency, True)
        elif entropy(self.Y[indices], class_frequency) == 0.0:
            return self.Node(depth, classes, class_frequency, True)
        gains = np.array(
            [continues_attribute_gain(self.X[indices], self.Y[indices], i) for i in range(self.X.shape[1])])
        target_att_id = gains[:, 0].argmax()
        # current_att_values = att_values[target_att_id]
        # current_att_values = get_value_frequency(self.X[indices, target_att_id])[0]
        threshold = gains[target_att_id, 1]
        children = []
        new_node = self.Node(depth, classes, class_frequency, False, target_att_id, children)
        child_indices = list(set([i for i, x in enumerate(self.X[:, target_att_id]) if x <= threshold]).
                             intersection(set(indices)))
        children.append((threshold, self.__build_node(new_node.vote, child_indices, depth + 1)))
        child_indices = list(set([i for i, x in enumerate(self.X[:, target_att_id]) if x > threshold]).
                             intersection(set(indices)))
        children.append((threshold, self.__build_node(new_node.vote, child_indices, depth + 1)))
        new_node.children = children
        return new_node

    def prune(self, validation_x, validation_y):
        self.__prune_node(self.root, validation_x, validation_y)

    def __prune_node(self, node, validation_x, validation_y):
        if node.is_leaf:
            return
        for child in node.children:
            self.__prune_node(child[1], validation_x, validation_y)
        before_score = self.accuracy_score(validation_x, validation_y)
        node.is_leaf = True
        after_score = self.accuracy_score(validation_x, validation_y)
        if after_score > before_score:
            print("node pruned!")
            node.children = None
            return
        node.is_leaf = False

    def predict(self, X):
        # print("test size: {}".format(X.shape[0]))
        predicts = []
        for i, x in enumerate(X):
            # if i % 100 == 0:
            #     print(i)
            node = self.root
            while not node.is_leaf:
                if x[node.att_index] <= node.children[0][0]:
                    node = node.children[0][1]
                else:
                    node = node.children[1][1]
            predicts.append(node.vote)
        return np.array(predicts, dtype=int)

    def accuracy_score(self, X, Y):
        predicted = self.predict(X)
        return accuracy_score(Y, predicted)

    def __str__(self):
        if self.root is None:
            return "None"
        return str(self.root)


class RandomForest:
    def __init__(self, n_estimators=10, min_sample_leaf_size=1):
        self.n_estimators = n_estimators
        self.min_sample_leaf_size = min_sample_leaf_size
        self.trees = [None for _ in range(n_estimators)]

    def fit(self, X, Y):
        for tree_id in range(self.n_estimators):
            new_data_set_ids = np.random.randint(0, X.shape[0], X.shape[0], dtype=int)
            new_x = X[new_data_set_ids]
            new_y = Y[new_data_set_ids]
            new_tree = DecisionTree(self.min_sample_leaf_size)
            new_tree.fit(new_x, new_y)
            self.trees[tree_id] = new_tree

    def predict(self, X):
        predicts = []
        votes = np.array([self.trees[i].predict(X) for i in range(self.n_estimators)]).T
        for v in votes:
            vote = np.argmax(np.bincount(v))
            predicts.append(vote)
        return np.array(predicts, dtype=int)


class GMM:
    def __init__(self, n_components, threshold=.01, max_iter=300):
        self.means = None
        self.covars = None
        self.pies = None
        self.gammas = None
        # self.classes = None
        self.k = n_components
        # self.threshold = .01
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X):
        k = self.k
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
            if 0 <= log_likelihood_changes < self.threshold:
                break
            if it > self.max_iter:
                break

            for i in range(X.shape[0]):
                for j in range(k):
                    self.gammas[j, i] = self.pies[j] * probs[j, i] / denominator[i]

            # maximization
            N = self.gammas.sum(axis=1)
            self.means = self.gammas.dot(X) / np.repeat(N, X.shape[1], axis=0).reshape(k, X.shape[1])
            for j in range(k):
                self.covars[j] = np.zeros((X.shape[1], X.shape[1]))
                for i in range(X.shape[0]):
                    xm = (X[i] - self.means[j]).reshape((X.shape[1], 1))
                    self.covars[j] += self.gammas[j, i] * xm.dot(xm.T)
                self.covars[j] /= N[j]
            self.pies = N / X.shape[0]
            it += 1
        print(self.means)
        print("converged in {} iterations!".format(it))

    def predict(self, X):
        k = self.k
        rv = [stats.multivariate_normal(self.means[j], self.covars[j], True) for j in range(k)]
        predicts = []
        for i, x in enumerate(X):
            cluster_id = np.argmax([self.pies[c_id] * rv[c_id].pdf(x) for c_id in range(k)])
            predicts.append(cluster_id)
        return predicts


class PCA:
    def __init__(self, n_components=10, features_are_less_than_samples=True):
        self.eigen_vectors = None
        self.features_are_less_than_samples = features_are_less_than_samples
        self.k = n_components

    def fit(self, x):
        if self.features_are_less_than_samples:
            self.__ordinary_fit(x)
        else:
            self.__tricky_fit(x)

    def __tricky_fit(self, x: np.ndarray):
        self.X = x
        mu = x.mean(axis=0)
        x_centered = x - mu
        eigen_values, eigen_vectors = eig(x_centered.dot(x_centered.T))
        eigen_vectors = x_centered.T.dot(eigen_vectors)
        norms = []
        for n in np.linalg.norm(eigen_vectors, axis=0):
            if n == 0:
                norms.append(1.0)
                continue
            norms.append(n)
        norms = np.array(norms)
        eigen_vectors = eigen_vectors / norms
        eigen_vectors = eigen_vectors.T
        eigen_values, eigen_vectors = zip(*sorted(zip(eigen_values, eigen_vectors), key=lambda t: t[0], reverse=True))
        eigen_vectors = np.array(eigen_vectors)
        self.eigen_vectors = np.array(eigen_vectors[:self.k]).T
        # print(reduced)

    def __ordinary_fit(self, x: np.ndarray):
        self.X = x
        mu = x.mean(axis=0)
        x_centered = x - mu
        eigen_values, eigen_vectors = eig(x_centered.T.dot(x_centered))
        # eigen_vectors = x_centered.T.dot(eigen_vectors)
        norms = []
        for n in np.linalg.norm(eigen_vectors, axis=0):
            if n == 0:
                norms.append(1.0)
                continue
            norms.append(n)
        norms = np.array(norms)
        eigen_vectors = eigen_vectors / norms
        eigen_vectors = eigen_vectors.T
        eigen_values, eigen_vectors = zip(*sorted(zip(eigen_values, eigen_vectors), key=lambda t: t[0], reverse=True))
        eigen_vectors = np.array(eigen_vectors)
        self.eigen_vectors = np.array(eigen_vectors[:self.k]).T
        # print(reduced)

    def transform(self, x):
        if type(x) is not np.ndarray:
            x = np.array(x)
        reduced = x.dot(self.eigen_vectors)
        return reduced

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class WindowBasedEnsembleClustering:
    def __init__(self, n_clusters, reduce_dimens, reduced_dimen_size, n_windows_in_row, n_windows_in_col, img_width,
                 img_height,
                 window_width_to_img_width,
                 window_height_to_img_height):
        self.n_clusters = n_clusters
        self.reduced_dimen_size = reduced_dimen_size
        self.n_windows_in_row = n_windows_in_row
        self.n_windows_in_col = n_windows_in_col
        self.reduce_dimens = reduce_dimens
        self.models = [[None for _ in range(n_windows_in_col)] for _ in range(n_windows_in_row)]
        self.dimen_reducers = [[None for _ in range(n_windows_in_col)] for _ in range(n_windows_in_row)]
        self.window_width = window_width_to_img_width * img_width
        self.window_height = window_height_to_img_height * img_height
        self.window_centers = [
            [(x, y) for x in np.linspace(self.window_width / 2, img_width - self.window_width / 2, n_windows_in_col)]
            for y in np.linspace(self.window_height / 2, img_height - self.window_height / 2, n_windows_in_row)]
        self.window_mappings = [[None for _ in range(n_windows_in_col)] for _ in range(n_windows_in_row)]

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
                            new_x.append(get_feature_from_kd_ds(kd, dss[img_id][m]))
                            new_y.append(Y[img_id])
                if len(new_x) == 0:
                    print("window {}; no data found!".format(window_id))
                    continue
                new_x = np.array(new_x)
                new_y = np.array(new_y)
                if new_x.shape[0] < 60 * 10 * self.n_clusters:
                    print("window {}; just {} data found!".format(window_id, new_x.shape[0]))
                    continue
                print("trainig window {} model with {} data!".format(window_id, new_x.shape[0]))
                if self.reduce_dimens and new_x.shape[0] > 1 and new_y.max() != new_y.min():
                    # dmr = PCA(n_components=.99, svd_solver='full')
                    dmr = PCA(n_components=self.reduced_dimen_size)
                    # dmr = LinearDiscriminantAnalysis(n_components=20)
                    dmr.fit(new_x)
                    new_x = np.array(dmr.transform(new_x))
                    self.dimen_reducers[i][j] = dmr
                # window_model = KM(self.n_clusters, verbose=True, n_jobs=2, max_iter=10, tol=10000)
                # window_model = KM(self.n_clusters, verbose=True, n_jobs=2)
                window_model = GaussianMixture(self.n_clusters, verbose=True)
                # window_classifier = KNeighborsClassifier(np.min([50, new_x.shape[0]]))
                # window_classifier = svm.SVC(C=np.power(10.0, -6), kernel='linear')
                window_model.fit(new_x)
                p = window_model.predict(new_x)
                print("window {} model trained with {} data!".format(window_id, new_x.shape[0]))
                print("window {} model purity: {}!".format(window_id, purity_score(p, new_y)))
                self.models[i][j] = window_model
        w_i, w_j = int(self.n_windows_in_row / 2), int(self.n_windows_in_col / 2)
        self.window_mappings[w_i][w_j] = [t for t in range(self.n_clusters)]
        self.find_mapping_of_window_clusters(w_i, w_j, kds, dss, Y)

    def find_mapping_of_window_clusters(self, win_i, win_j, kds, dss, Y):
        adjacent_window_ids = []
        for i in range(self.n_windows_in_row):
            for j in range(self.n_windows_in_col):
                if self.windows_have_overlapping_area(win_i, win_j, i, j):
                    if not (self.models[i][j] is None or self.window_mappings[i][j] is not None):
                        adjacent_window_ids.append((i, j))
        windows_with_no_intersection = []
        for w_id, (i, j) in enumerate(adjacent_window_ids):
            shared_x = []
            shared_y = []
            for img_id in range(kds.shape[0]):
                for m, kd in enumerate(kds[img_id]):
                    if self.is_point_in_window(kd.pt, win_i, win_j) and self.is_point_in_window(kd.pt, i, j):
                        shared_x.append(get_feature_from_kd_ds(kd, dss[img_id][m]))
                        shared_y.append(Y[img_id])
            if len(shared_x) == 0:
                print("window {} didn't have intersection with {}".format((i, j), (win_i, win_j)))
                windows_with_no_intersection.append(w_id)
                # self.models[i][j] = None
                continue
            if self.reduce_dimens and self.dimen_reducers[win_i][win_j] is not None:
                shared_ref_x = np.array(self.dimen_reducers[win_i][win_j].transform(shared_x))
            else:
                shared_ref_x = shared_x
            if self.reduce_dimens and self.dimen_reducers[i][j] is not None:
                shared_target_x = np.array(self.dimen_reducers[i][j].transform(shared_x))
            else:
                shared_target_x = shared_x
            ref_prediction = self.models[win_i][win_j].predict(shared_ref_x)
            target_prediction = self.models[i][j].predict(shared_target_x)
            ref_clusters_freq = np.bincount(ref_prediction)
            target_clusters_freq = np.bincount(target_prediction)
            self.window_mappings[i][j] = [None for _ in range(self.n_clusters)]
            temp_sorted_ref_clusters_freq = np.sort(ref_clusters_freq)
            temp_sorted_target_clusters_freq = np.sort(target_clusters_freq)
            for t in range(np.min([ref_clusters_freq.shape[0], target_clusters_freq.shape[0]])):
                ref_max_arg = np.where(ref_clusters_freq == temp_sorted_ref_clusters_freq[-int(t + 1)])[0][0]
                target_max_arg = np.where(target_clusters_freq == temp_sorted_target_clusters_freq[-int(t + 1)])[0][0]
                if ref_clusters_freq[ref_max_arg] == 0 or target_clusters_freq[target_max_arg] == 0:
                    break
                self.window_mappings[i][j][target_max_arg] = self.window_mappings[win_i][win_j][ref_max_arg]
        adjacent_window_ids = np.delete(adjacent_window_ids, windows_with_no_intersection, axis=0)
        for i, j in adjacent_window_ids:
            self.find_mapping_of_window_clusters(i, j, kds, dss, Y)

    def windows_have_overlapping_area(self, win_i, win_j, i, j):
        x1s = self.window_centers[win_i][win_j][0] - self.window_width / 2
        x1e = self.window_centers[win_i][win_j][0] + self.window_width / 2
        y1s = self.window_centers[win_i][win_j][1] - self.window_height / 2
        y1e = self.window_centers[win_i][win_j][1] + self.window_height / 2

        x2s = self.window_centers[i][j][0] - self.window_width / 2
        x2e = self.window_centers[i][j][0] + self.window_width / 2
        y2s = self.window_centers[i][j][1] - self.window_height / 2
        y2e = self.window_centers[i][j][1] + self.window_height / 2

        if (x2s <= x1s <= x2e or x2s <= x1e <= x2e) and (y2s <= y1s <= y2e or y2s <= y1e <= y2e):
            return True
        return False

    def predict(self, kds, dss):
        predicted = []
        for img_id in range(kds.shape[0]):
            votes = []
            for i in range(self.n_windows_in_row):
                for j in range(self.n_windows_in_col):
                    if self.models[i][j] is None or self.window_mappings[i][j] is None:
                        continue
                    new_x = []
                    for m, kd in enumerate(kds[img_id]):
                        if self.is_point_in_window(kd.pt, i, j):
                            new_x.append(get_feature_from_kd_ds(kd, dss[img_id][m]))
                    if len(new_x) > 0:
                        if self.reduce_dimens and self.dimen_reducers[i][j] is not None:
                            new_x = np.array(self.dimen_reducers[i][j].transform(new_x))
                        votes.extend(
                            [self.window_mappings[i][j][p] for p in self.models[i][j].predict(np.array(new_x)) if
                             self.window_mappings[i][j][p] is not None])
            vote = np.argmax(np.bincount(votes))
            predicted.append(vote)
        return predicted


class WindowBasedEnsembleClassifier:
    def __init__(self, reduce_dimens, reduced_dimens, criterion, n_windows_in_row, n_windows_in_col, img_width,
                 img_height,
                 window_width_to_img_width,
                 window_height_to_img_height, n_estimators=10):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.n_windows_in_row = n_windows_in_row
        self.n_windows_in_col = n_windows_in_col
        self.reduce_dimens = reduce_dimens
        self.reduced_dimens = reduced_dimens
        self.classifiers = [[None for _ in range(n_windows_in_col)] for _ in range(n_windows_in_row)]
        self.dimen_reducers = [[None for _ in range(n_windows_in_col)] for _ in range(n_windows_in_row)]
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
                            new_x.append(get_feature_from_kd_ds(kd, dss[img_id][m]))
                            new_y.append(Y[img_id])
                if len(new_x) == 0:
                    print("window {}; no data found!".format(window_id))
                    continue
                new_x = np.array(new_x)
                new_y = np.array(new_y)
                # if new_y.max() == new_y.min():
                #     print("window {}; just one class!".format(window_id))
                #     continue
                print("trainig window {} classifier with {} data!".format(window_id, new_x.shape[0]))
                if self.reduce_dimens and new_x.shape[0] > 1 and new_y.max() != new_y.min():
                    # dmr = PCA(n_components=.99, svd_solver='full')
                    dmr = PCA(n_components=self.reduced_dimens)
                    # dmr = LinearDiscriminantAnalysis(n_components=20)
                    dmr.fit(new_x)
                    new_x = np.array(dmr.transform(new_x))
                    self.dimen_reducers[i][j] = dmr
                window_classifier = RandomForestClassifier(self.n_estimators, criterion=self.criterion)
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
                            new_x.append(get_feature_from_kd_ds(kd, dss[img_id][m]))
                    if len(new_x) > 0:
                        if self.reduce_dimens and self.dimen_reducers[i][j] is not None:
                            new_x = np.array(self.dimen_reducers[i][j].transform(new_x))
                        votes.extend(self.classifiers[i][j].predict(np.array(new_x)))
            vote = np.argmax(np.bincount(votes))
            predicted.append(vote)
        return np.array(predicted)


if __name__ == '__main__':
    train_kds, train_dss, train_labels = extract_features()
    model = DecisionTree(10)
    new_x = []
    new_y = []
    for img_id in range(train_kds.shape[0]):
        for m, kd in enumerate(train_kds[img_id]):
            new_x.append(get_feature_from_kd_ds(kd, train_dss[img_id][m]))
            new_y.append(train_labels[img_id])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    model.fit(new_x, new_y)
