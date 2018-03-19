import math
import random

from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(
    cancer_dataset['data'], cancer_dataset['target'], random_state=0)



'''from sklearn.model_selection import train_test_split
from sklearn import datasets
data, target = datasets.make_regression(n_features=2,
                                 shuffle=True, random_state=2)
train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.3)'''

class Neuron():

    def __init__(self, Layer, NeuronsNum = 0):
        self.layer = Layer
        self.value = 0
        self.d = 0
        self.input = []
        if self.layer != 0:
            for i in range(0, NeuronsNum):
                self.input.append(random.random())





class Neuron_network():

    def __init__(self, NumOfLayers, PerceptNum, NumOfFeatures, y_train):
        self.alpha = 0.0001
        self.middle = []
        self.dispersion = []
        self.classes = self.FindClasses(y_train)
        self.PerceptNum = PerceptNum
        self.NumOfLayers = NumOfLayers
        self.NeuronsMas = []
        for i in range(0, self.NumOfLayers + 2):
            self.NeuronsMas.append([])
            if i == 0:
                for j in range(0, NumOfFeatures):
                    self.NeuronsMas[i].append(Neuron(i))
            elif i != self.NumOfLayers + 1:
                for j in range(0, self.PerceptNum):
                    self.NeuronsMas[i].append(Neuron(i, len(self.NeuronsMas[i-1])))
            elif i == self.NumOfLayers + 1:
                for j in range(0, len(self.classes)):
                    self.NeuronsMas[i].append(Neuron(i, len(self.NeuronsMas[i-1])))


    def FindClasses(self, mas):
        cl = {}
        lenght = 0
        for i in range(0, len(mas)):
            add = True
            for j in range(0, len(cl)):
                if cl.get(j) == mas[i]:
                    add = False
                    break
            if add:
                cl[lenght] = mas[i]
                lenght += 1
        return cl


    def nonlin(self, x, deriv=False):
        if x < -500:
            fx = 0
        else:
            fx = 1 / (1 + pow(math.e, (x * (-1) * self.alpha)))
        if(deriv == True):
            return fx*(1-fx)
        return fx

    def maxNeuronValue(self, mas):
        max = [0, mas[0]]
        for i in range(0, len(mas)):
            if mas[i] > max[1]:
                max = [i, mas[0]]
        return max[0]

    def XnewNormalize(self, mas):
        for i in range(0, len(mas)):
            mas[i] = (mas[i] - self.middle[i])/self.dispersion[i]
        return mas

    def MinMaxNormalize(self, mas):
        for i in range(0, len(mas[0])):
            self.middle.append(0)
            for j in range(0, len(mas)):
                self.middle[i] += mas[j][i]
            self.middle[i] = self.middle[i]/len(mas)
            self.dispersion.append(0)
            for j in range(0, len(mas)):
                self.dispersion[i] += pow((mas[j][i] - self.middle[i]), 2)
            self.dispersion[i] = pow(self.dispersion[i] / (len(mas) - 1), 1/2)

    def predict(self, X_new):
        #X_new = self.XnewNormalize(X_new)
        for i in range(0, len(self.NeuronsMas)):
            if i == 0:
                for j in range(len(self.NeuronsMas[i])):
                    self.NeuronsMas[i][j].value = X_new[j]
            else:
                for j in range(0, len(self.NeuronsMas[i])):
                    for k in range(0, len(self.NeuronsMas[i-1])):
                        self.NeuronsMas[i][j].value += self.NeuronsMas[i-1][k].value * self.NeuronsMas[i][j].input[k]
                    self.NeuronsMas[i][j].value = self.nonlin(self.NeuronsMas[i][j].value)
        valueMas = []
        for i in range(0, len(self.NeuronsMas[len(self.NeuronsMas)-1])):
            valueMas.append(self.NeuronsMas[len(self.NeuronsMas)-1][i].value)
        return self.classes.get(self.maxNeuronValue(valueMas))


    def learn(self, X_train, y_train, alpha = 0.0001, iterationLimit = 10000):
        self.alpha = alpha
        #self.MinMaxNormalize(X_train)
        for iter in range(0, iterationLimit):
            for i in range(0, len(X_train)):
                y_predicted = self.predict(X_train[i])
                for j in range(0, len(self.NeuronsMas[len(self.NeuronsMas)-1])):
                    if self.classes.get(j) != y_train[i]:
                        y_true = 0
                    else:
                        y_true = 1
                    self.NeuronsMas[len(self.NeuronsMas) - 1][j].d = 0
                    self.NeuronsMas[len(self.NeuronsMas)-1][j].d = y_true - self.NeuronsMas[len(self.NeuronsMas)-1][j].value

                for j in range(len(self.NeuronsMas) - 2, 0, -1):
                    for k in range(0, len(self.NeuronsMas[j])):
                        self.NeuronsMas[j][k].d = 0
                        for m in range(0, len(self.NeuronsMas[j+1])):
                            self.NeuronsMas[j][k].d += self.NeuronsMas[j+1][m].d * self.NeuronsMas[j+1][m].input[k]

                for j in range(1, len(self.NeuronsMas)):
                    for k in range(0, len(self.NeuronsMas[j])):
                        interСalc = alpha * self.NeuronsMas[j][k].d * self.nonlin(self.NeuronsMas[j][k].value, True)
                        for m in range(0, len(self.NeuronsMas[j][k].input)):
                            self.NeuronsMas[j][k].input[m] = self.NeuronsMas[j][k].input[m] + interСalc * self.NeuronsMas[j-1][m].value


'''train_data = [[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]]

# выходные данные
train_labels = [0, 1, 1, 0]'''

newNetwork = Neuron_network(2, 100, len(train_data[0]), train_labels)
newNetwork.learn(train_data, train_labels, alpha=1,iterationLimit=100)

divide = [0, 0]
for i in range(0, len(train_data)):
    if newNetwork.predict(train_data[i]) == train_labels[i]:
        divide[0] += 1
    else:
        divide[1] += 1

print("Правильность на обучающей выборке: ", divide[0]/len(train_labels))

divide = [0, 0]
for i in range(0, len(test_data)):
    if newNetwork.predict(test_data[i]) == test_labels[i]:
        divide[0] += 1
    else:
        divide[1] += 1

print("Правильность на тестовой выборке: ", divide[0]/len(test_labels))
