from sklearn.model_selection import train_test_split
from sklearn import datasets
data, target = datasets.make_regression(n_features=6,
                                 shuffle=True, random_state=2)
train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.3)

class Linear_regression():

    def __init__(self):
        self.KoeffMas = []

    def Gauss(self, Matrix, Answers, n):

        for i in range(0, n):
            self.KoeffMas.append(0)

        for i in range(0, len(Matrix)):
            for j in range(0, len(Matrix)):
                if j == i:
                    continue
                Kdel = Matrix[j][i] / Matrix[i][i]
                for k in range(0, len(Matrix[0])):
                    Matrix[j][k] -= Matrix[i][k] * Kdel
                Answers[j] -= Answers[i] * Kdel

        for i in range(0, len(Answers)):
            if Matrix[i][i] == 0:
                self.KoeffMas[i] = 'Null'
            else:
                self.KoeffMas[i] = Answers[i] / Matrix[i][i]


    def learn(self, X_train, y_train):
        XSumMas = []
        YSumMas = []

        for i in range(0, len(X_train[0])):
            XSumMas.append([])
            YSumMas.append(0)
            for j in range(0, len(X_train[0])):
                XSumMas[i].append(0)
                for k in range(0, len(X_train)):
                    if i != len(X_train[0]):
                        if j == len(X_train[0]):
                            XSumMas[i][j] += X_train[k][i]
                        else:
                            XSumMas[i][j] += X_train[k][i] * X_train[k][j]
                    if i == len(X_train[0]):
                        if j == len(X_train[0]):
                            XSumMas[i][j] = len(X_train)
                        else:
                            XSumMas[i][j] += X_train[k][j]

        for i in range(0, len(X_train[0])):
            for j in range(0, len(X_train)):
                if i != len(X_train[0]):
                    YSumMas[i] += y_train[j] * X_train[j][i]
                else:
                    YSumMas[i] += y_train[j]

        self.Gauss(XSumMas, YSumMas, len(X_train[0])+1)

    def predict(self, X_new):
        regression_answer = 0
        for i in range(0, len(X_new)):
            if self.KoeffMas == "Null":
                continue
            else:
                regression_answer += self.KoeffMas[i] * X_new[i]
        return regression_answer



