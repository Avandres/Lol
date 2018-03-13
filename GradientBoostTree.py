from TreeClassifier import tree_classifier as tree
import math

class Gradient_boost_forest():

    def __init__(self, i):
        self.iteration = i
        self.forest= []
        self.est_weights = []


    def predict_test(self, estimator, X_test, y_test, weights):
        Et = 0
        for i in range(0, len(X_test)):
            if estimator.predict(X_test[i]) != y_test[i]:
                Et += weights[i]
        return Et


    def learn(self, X_train, y_train, tree_size, alpha=0.5):
        obj_weights = []
        for i in range(0, len(X_train)):
            obj_weights.append(1/len(X_train))

        for i in range(0, self.iteration):
            print(i)
            real_X_train = []
            real_y_train = []
            for j in range(0, len(X_train)):
                w = 0
                while w < obj_weights[j]:
                    real_X_train.append(X_train[j])
                    real_y_train.append(y_train[j])
                    w += 1/len(X_train)
            self.forest.append(tree())
            self.forest[i].learn(real_X_train, real_y_train, tree_size)
            Et = self.predict_test(self.forest[i], X_train, y_train, obj_weights)
            self.est_weights.append(alpha * math.log((1-Et)/Et)/2)
            w0 = 0
            for j in range(0, len(X_train)):
                if self.forest[i].predict(X_train[j]) != y_train[j]:
                    obj_weights[j] *= math.exp(self.est_weights[i] * y_train[j] * self.forest[i].predict(X_train[j]))
                    w0 += obj_weights[j]




    def most_frequent(self, mas):
        '''Функция возвращает наиболее часто встречающийся
        класс из имеющихся в массиве'''
        clMas = []
        for i in range(0, len(mas)):
            add = 1
            for j in range(0, len(clMas)):
                if mas[i][1] == clMas[j][1]:
                    add = 0
                    clMas[j][0] += mas[i][0]
                    break
            if add == 1:
                clMas.append([mas[i][0], mas[i][1]])

        max = clMas[0]
        for i in range(0, len(clMas)):
            if max[0] < clMas[i][0]:
                max = clMas[i]

        return max[1]


    def predict(self, X_new):
        answ_mas = []
        for i in range(0, len(self.forest)):
            answ_mas.append([self.est_weights[i], self.forest[i].predict(X_new)])
        answer = self.most_frequent(answ_mas)
        return answer


from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer_dataset['data'], cancer_dataset['target'], random_state=0)


newGBF = Gradient_boost_forest(10)
newGBF.learn(X_train, y_train, 2)

divide = [0, 0]
for i in range(0, len(X_test)):
    if newGBF.predict(X_test[i]) == y_test[i]:
        divide[0] += 1
    else:
        divide[1] += 1

print(divide[0]/len(y_test))



