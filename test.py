import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
plt.rc("font", family = "Verdana")
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer_dataset['data'], cancer_dataset['target'], random_state=0)

from sklearn.ensemble import gradient_boosting
RF = gradient_boosting.GradientBoostingClassifier(n_estimators=10)

RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)
print(y_pred, "\n")
print("Правильность: ", RF.score(X_test, y_test))





