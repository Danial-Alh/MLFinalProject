import numpy as np
from sklearn.metrics.classification import accuracy_score


def get_value_frequency(data):
    classes = np.unique(data)
    class_frequency = [0 for _ in classes]
    for d in data:
        for i, c in enumerate(classes):
            if d == c:
                class_frequency[i] += 1
    return np.array(classes), np.array(class_frequency, dtype=float)


def entropy(Y):
    class_frequency = get_value_frequency(Y)[1]
    p = class_frequency / Y.shape[0]
    return - (p * np.log2(p)).sum()


def gain(X, Y, att_index):
    classes, value_sizes = get_value_frequency(X[:, att_index])
    entropies = np.array([entropy(np.array([Y[i] for i, x in enumerate(X) if x[att_index] == c])) for c in classes])
    return entropy(Y) - (value_sizes / X.shape[0] * entropies).sum()


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

    def __init__(self):
        self.X = None
        self.Y = None
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
        if depth + 1 == self.X.shape[1]:
            return self.Node(depth, classes, class_frequency, True)
        elif entropy(self.Y[indices]) == 0.0:
            return self.Node(depth, classes, class_frequency, True)
        gains = np.array([gain(self.X[indices], self.Y[indices], i) for i in range(self.X.shape[1])])
        target_att_id = gains.argmax()
        # current_att_values = att_values[target_att_id]
        current_att_values = get_value_frequency(self.X[indices, target_att_id])[0]
        children = []
        new_node = self.Node(depth, classes, class_frequency, False, target_att_id, children)
        for v in current_att_values:
            child_indices = list(set([i for i, x in enumerate(self.X[:, target_att_id]) if x == v]).
                                 intersection(set(indices)))
            children.append((v, self.__build_node(new_node.vote, child_indices, depth + 1)))
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
                found = False
                for child in node.children:
                    if child[0] == x[node.att_index]:
                        node = child[1]
                        found = True
                        break
                if not found:
                    break

            predicts.append(node.vote)
        return np.array(predicts, dtype=int)

    def accuracy_score(self, X, Y):
        predicted = self.predict(X)
        return accuracy_score(Y, predicted)

    def __str__(self):
        if self.root is None:
            return "None"
        return str(self.root)
