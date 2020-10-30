#Proyecto implementación de algoritmo KNN
#Rodrigo Morales Aguayo A01632834

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
