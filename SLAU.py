def Gauss(Matrix, Answers, n):
    A = []
    for i in range(0, n):
        A.append(0)

    for i in range(0, n-1):
        for j in range(i+1, n):
            K = Matrix[j][i]/Matrix[i][i]
            for l in range(0, n):
                Matrix[j][l] = Matrix[j][l]+Matrix[i][l]*(-1)*K
            Answers[j] = Answers[j]+Answers[i]*(-1)*K
    A[n-1] = Answers[n-1]/Matrix[n-1][n-1]
    K = A[n-1] * Matrix[n - 2][n-1]
    for i in range(n-2, -1, -1):
        A[i] =(Answers[i]-K)/Matrix[i][i]
        if i>0:
            K = 0
            for j in range(i, n):
                K += A[j]*Matrix[i-1][j]

    return (A)