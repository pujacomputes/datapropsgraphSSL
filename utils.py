import torch
import copy
from edge_removing_caa import EdgeRemovingCAA 
from edge_removing_gga import EdgeRemovingGGA
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

def load_datasets(dataset_list):
    NUM_DATASETS = len(dataset_list)
    all_data = []
    for e_num, d_name in enumerate(dataset_list):
        dataset = torch.load(d_name)
        for d in dataset:
            d.y = e_num
            d.edge_attr = None
            d.x = torch.ones(d.num_nodes, 10)
        all_data += dataset
    print("=> Total Samples: ", len(all_data))
    print("=> Num Classes: ", NUM_DATASETS)
    return all_data

class CustomTransform:
    def __init__(self, aug1, aug2, eval=False):
        self.aug0 = torch.nn.Identity()
        self.aug1 = aug1
        self.aug2 = aug2
        self.eval = eval

    def __call__(self, x):
        if self.eval:
            x1 = self.aug0(copy.deepcopy(x))
        else:
            x1 = self.aug1(copy.deepcopy(x))
        x2 = self.aug2(copy.deepcopy(x))
        return x, x1, x2


def get_augmentor(args):
    if args.aug_type == "gga":
        aug1 = torch.nn.Identity()
        aug2 = EdgeRemovingGGA(pe=args.aug_ratio)
    elif args.aug_type == "gga_sym":
        aug1 = EdgeRemovingGGA(pe=args.aug_ratio)
        aug2 = EdgeRemovingGGA(pe=args.aug_ratio)

    elif args.aug_type == "caa":
        aug1 = torch.nn.Identity()
        aug2 = EdgeRemovingCAA(pe=args.aug_ratio)

    elif args.aug_type == "caa_sym":
        aug1 = EdgeRemovingCAA(pe=args.aug_ratio)
        aug2 = EdgeRemovingCAA(pe=args.aug_ratio)
    else:
        print("ERROR Invalid Augmentation")
        exit()
    cust_aug = CustomTransform(aug1=aug1, aug2=aug2)
    return cust_aug

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring="accuracy", verbose=0
            )
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring="accuracy", verbose=0
            )
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    test_acc = accuracy_score(y_test, classifier.predict(x_test))
    train_acc = accuracy_score(y_train, classifier.predict(x_train))

    results = {
        "micro_f1": -1,
        "macro_f1": -1,
        "train_acc": np.mean(accuracies),
        "valid_acc": -1,
        "test_acc": np.mean(accuracies_val),
    }
    return results

def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring="accuracy", verbose=0
            )
        else:
            classifier = LinearSVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring="accuracy", verbose=0
            )
        else:
            classifier = LinearSVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    test_acc = accuracy_score(y_test, classifier.predict(x_test))
    train_acc = accuracy_score(y_train, classifier.predict(x_train))

    results = {
        "micro_f1": -1,
        "macro_f1": -1,
        "train_acc": np.mean(accuracies),
        "valid_acc": -1,
        "test_acc": np.mean(accuracies_val),
    }
    return results