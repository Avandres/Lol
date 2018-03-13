import TreeClassifier
import random
import copy
import numpy as np
import math

random.seed()

from sklearn.datasets import load_breast_cancer

cancer_dataset = load_breast_cancer()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer_dataset['data'], cancer_dataset['target'], random_state=0)


class Stacking_tree_classifier():

    def __init__(self, numbOfTree):
        self.forest = []
        for i in range(0, numbOfTree):
            self.forest.append(TreeClassifier.tree_classifier())
        self.metaClassifier = TreeClassifier.tree_classifier()
        self.MetaX_train = []

    def is_only_one(self, mas):
        answ = 1
        value = mas[0]
        for i in range(0, len(mas)):
            if mas[i] != value:
                answ = 0
                break
        return answ

    def shiffle(self, X_train, y_train):
        new_X_train = []
        new_y_train = []
        for i in range(0, len(X_train)):
            rand = random.randint(0, len(X_train) - 1)
            new_X_train.append([])
            new_y_train.append([])
            new_X_train[i] = copy.copy(X_train[rand])
            new_y_train[i] = copy.copy(y_train[rand])
        return new_X_train, new_y_train

    def learn_forest(self, X_train, y_train, tree_size):
        for i in range(0, len(self.forest)):
            train = self.shiffle(X_train, y_train)
            self.forest[i].learn(train[0], train[1], tree_size)
            print(i)
        for i in range(0, len(X_train)):
            self.MetaX_train.append([])
            for j in range(0, len(self.forest)):
                self.MetaX_train[i].append(self.forest[j].predict(X_train[i]))
        self.metaClassifier.learn(self.MetaX_train, y_train, tree_size)

    def predict(self, X_new):
        X = []
        for i in range(0, len(self.forest)):
            X.append(self.forest[i].predict(X_new))
        return self.metaClassifier.predict(X)


new_forest = Stacking_tree_classifier(5)
new_forest.learn_forest(X_train, y_train, 20)

divide = [0, 0]
for i in range(0, len(X_test)):
    if new_forest.predict(X_test[i]) == y_test[i]:
        divide[0] += 1
    else:
        divide[1] += 1

print(divide[0] / len(y_test))
