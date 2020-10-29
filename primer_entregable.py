import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from random import randrange as rr
import random


def genData(n, seed=1, m=False):  # Genera data linearmente separable por una pendiente (m)
    np.random.seed(seed)
    if (m == False):
        m = (rr(20) + 1) / 10

    X = np.random.random((n, 2))
    y = ((X[:, 1] / X[:, 0]) > m) * 1
    X = (X - 0.5) * 2
    return [X, y]


def graphPoints(X, y):  # Grafica los puntos y los puntos proyectados
    for i in range(0, X.shape[0]):
        p = X[i, 0:2]
        if (y[i]):
            plt.scatter(p[0], p[1], color="red", s=4)
        else:
            plt.scatter(p[0], p[1], color="black", s=4)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


def randomVector():
    v = [random.randint(-9, 9), random.randint(-9, 9)]
    v = np.array(v)
    return v


def drawVector(lv):  # list of vectors
    maxV = 0
    colors = ['black', 'red', 'green', 'orange', 'grey', 'purple', 'brown', 'purple']
    for i in range(len(lv)):
        v = lv[i]
        plt.quiver(0, 0, v[0], v[1], color=colors[i % len(colors)], angles='xy', scale_units='xy', scale=1)
        if (max(v) > maxV):
            maxV = max(v)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


def projuv(u, v):
    uv = u.dot(v)
    nv = la.norm(v)
    comp = uv / (nv ** 2)
    return comp * v


def pointProyections(X, y,
                     v):  # pro = producto punto de vectores sobre la magnitud del segundo al cuadrado por el segundo vector
    Xp = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        p = X[i, 0:2]
        Xp[i, :] = projuv(p, v)
    return Xp


[X, y] = genData(100)
graphPoints(X, y)

for i in range(100):
    v = randomVector()
    drawVector([v])
    Xp = pointProyections(X, y, v)
    graphPoints(Xp, y)
