import math

class GaussianNB():

    def __init__(self):
        pass

    def standart_deviation_and_xmid(self, X_train, j):
        x_mid = 0
        for i in range(0, len(X_train)):
            x_mid += X_train[i][j]
        x_mid = x_mid/len(X_train)
        x_sum = 0
        for i in range(0, len(X_train)):
            x_sum += pow((X_train[i][j] - x_mid), 2)
        deviation = pow((x_sum/len(X_train)), 1/2)
        return [deviation, x_mid]

    def normal_function(self, dev, a, x):
        e = pow(math.e, -(pow(x-a, 2)/(2*pow(dev, 2))))
        f = e/(math.sqrt(2*math.pi) * dev)
        return f

    def something(self, dev, a):
        maxf = self.normal_function(dev, a, a)
        sigma_point = self.normal_function(dev, a, a+dev)
        sigma2_point = self.normal_function(dev, a, a+2*dev) 


