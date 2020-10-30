# Proyecto: implementación de algoritmo KNN
# Oscar Miranda Escalante A01630791 (oscarmires)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mode

def calc_distances(x0, y0, X):
    # genera un arreglo con las distancias entre un punto y todos los puntos de X
    distances = []
    for i in range(X.shape[0]):
        point = X[i, 0:2]
        distances.append(euclidean_distance(x0, y0, point[0], point[1]))
    return np.array(distances)

def clasificacion_knn(X, y, px, py, k=False):
    # devuelve la clase a la que pertenece el punto
    if not k:
        k = int(len(X) ** 0.5)
    nearest_neighbors = calc_distances(px, py, X).argsort()[0:k]
    mode_knn = mode(y[nearest_neighbors])
    print("Nearest neighbors' indexes:", nearest_neighbors)
    print("Moda: ", mode_knn)
    plt.scatter(px, py, color="black", marker='^', s=30)
    for i in range(k):
        plt.scatter(X[nearest_neighbors[i], 0], X[nearest_neighbors[i], 1], color="yellow", s=20)
    graphPoints(X, y)
    return mode_knn
