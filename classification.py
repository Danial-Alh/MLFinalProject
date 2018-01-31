import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from feature_extractor import extract_features
from models import WindowBasedEnsembleClassifier


def parameter_tuner():
    train_kds, train_dss, train_labels = extract_features()
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
    kf = KFold(n_splits=3, shuffle=True)
    accuracies = []
    pars = ['entropy', 'gini']
    for par in pars:
        print(par)
        r = []
        for train_index, test_index in kf.split(train_kds):
            print("TRAIN:", train_index, "TEST:", test_index)
            kds_temp_train, dss_temp_train, kds_temp_test, dss_temp_test = \
                train_kds[train_index], train_dss[train_index], train_kds[test_index], train_dss[test_index]
            y_temp_train, y_temp_test = train_labels[train_index], train_labels[test_index]
            model = WindowBasedEnsembleClassifier(False, 40, par, 8, 8, 192, 192, 1.0 / 7, 1.0 / 7, 20)
            model.fit(kds_temp_train, dss_temp_train, y_temp_train)
            predicted = model.predict(kds_temp_test, dss_temp_test)
            acc = accuracy_score(y_temp_test, predicted)
            r.append(acc)
            print("total accuracy: {}".format(acc))
            for i in range(10):
                t_ids = [m for m, t in enumerate(y_temp_test) if t == i]
                t_acc = accuracy_score(y_temp_test[t_ids], predicted[t_ids])
                print("number {} accuracy: {}".format(i, t_acc))
        r = np.array(r)
        accuracies.append(r.mean())
        print("after")
    acc_max_arg = np.argmax(accuracies)
    print("best {}: {}".format(pars[acc_max_arg], accuracies[acc_max_arg]))
    for a, c in zip(accuracies, pars):
        print("{}: {}".format(c, a))
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


def final_train():
    train_kds, train_dss, train_labels = extract_features()
    test_kds, test_dss, test_labels = extract_features('testing')
    accuracies = []
    for _ in range(10):
        model = WindowBasedEnsembleClassifier(False, 40, 'entropy', 8, 8, 192, 192, 1.0 / 7, 1.0 / 7, 20)
        model.fit(train_kds, train_dss, train_labels)
        predicted = model.predict(test_kds, test_dss)
        acc = accuracy_score(test_labels, predicted)
        accuracies.append(acc)
        print("total accuracy: {}".format(acc))
        for i in range(10):
            t_ids = [m for m, t in enumerate(test_labels) if t == i]
            t_acc = accuracy_score(test_labels[t_ids], predicted[t_ids])
            print("number {} accuracy: {}".format(i, t_acc))
    print("mean accuracy: {}".format(np.mean(accuracies)))


if __name__ == '__main__':
    # parameter_tuner()
    final_train()
