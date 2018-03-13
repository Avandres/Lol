import SLAU

x = []
y = []

n = input("введите количество переменных: ")
n = int(n)

m = input("Введите количество опытов: ")
m = int(m)

for i in range(0, m):
    y.append(0)

for i in range(0, n):
    x.append([])
    for j in range(0, m):
        x[i].append(0)

for i in range(0, n):
    for j in range(0, m):
        title = "Введите значение переменной " + str(i+1) + " в " + str(j+1) + "-ом опыте: "
        x[i][j] = input(title)

for i in range(0, m):
    title = "Введите значение переменной y в " + str(i+1) + "-ом опыте: "
    y[i] = input(title)



def Linear(X, Y, n, m):

    Xsum = []
    XYsum = []

    for i in range(0, n+1):
        Xsum.append([])
        XYsum.append(0)
        for j in range(0, n+1):
            Xsum[i].append(0)

    for i in range(0, n+1):
        for j in range(0, m):
            if i != n:
                XYsum[i] += float(Y[j])*float(X[i][j])
            else:
                XYsum[i] += float(Y[j])

        for k1 in range(0, n+1):
            for k2 in range(0, m):
                if i != n:
                    if k1 == n:
                        Xsum[i][k1] += float(X[i][k2])
                    else:
                        Xsum[i][k1] += float(X[i][k2])*float(X[k1][k2])
                if i == n:
                    if k1 == n:
                        Xsum[i][k1] = m
                    else:
                        Xsum[i][k1] += float(X[k1][k2])


    Answ = SLAU.Gauss(Xsum, XYsum, n+1)
    print(Answ)



Linear(x, y, n, m)