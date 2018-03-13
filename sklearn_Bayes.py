import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
plt.rc("font", family = "Verdana")
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Ключи датасета:\n", format(iris_dataset.keys()))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                        hist_kwds={'bins':20}, s=60, alpha=8, cmap=mglearn.cm3)

from sklearn.naive_bayes import BernoulliNB
bayes = BernoulliNB()

bayes.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = bayes.predict(X_new)
print("\nПрогноз: ", format(prediction))
print("\nОтвет: ", iris_dataset['target_names'][prediction])

y_pred = bayes.predict(X_test)
print(y_pred, "\n")
print("Правильность: ", bayes.score(X_test, y_test))