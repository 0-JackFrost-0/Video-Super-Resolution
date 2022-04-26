import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def generate_dataset1():
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]


def generate_dataset2():
    x = [np.random.rand() for i in range(1000)]
    y = [(0.5 - x[i])*(0.7 - x[i]) + 1 for i in range(1000)]
    return [x, y]


def max_eig(e):
    maxi = e[0]
    index = 0
    for i in range(len(e)):
        if maxi < e[i]:
            maxi = e[i]
            index = i
    return [maxi, index]


[x, y] = generate_dataset1()
[x2, y2] = generate_dataset2()

x_m = np.mean(x)
y_m = np.mean(y)
X = np.column_stack((x, y))

C = np.cov(X.T)
e, V = np.linalg.eig(C)

max_eig, index = max_eig(e)

u = V[:, index]
Z = np.dot(X, u)
### Havent centered the dataset yet, i.e. shift data keeping centroid at the origin, then it'll be fixed
Y = np.outer(Z, u)
plt.scatter(X[:, 0], X[:, 1])
plt.axline(Y.T[:, 0] + [x_m, y_m], Y.T[:, 1] + [x_m, y_m], color='orange')
plt.show()
