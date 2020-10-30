# Proyecto: implementación de algoritmo KNN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mode

# Oscar Miranda Escalante A01630791 (oscarmires)
def clasificacion_knn(X, y, px, py, k=False):
    # Devuelve la clase a la que pertenece el punto
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

# Rodrigo Morales Aguayo A01632834 (ROmorales08)
def calc_distances(x0, y0, X):
    # Genera un arreglo con las distancias entre un punto y todos los puntos de X
    distances = []
    for i in range(X.shape[0]):
        point = X[i, 0:2]
        distances.append(euclidean_distance(x0, y0, point[0], point[1]))
    return np.array(distances)

# Marian Alejandra Herrera Ayala (saalej)
df = pd.read_csv("iris1.csv") # Importar datos

# Adecuar datos
clases = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]
df["Tipo_Flor"] = df["Tipo_Flor"].replace(clases, [0, 1, 2])
data = df.values
X = data[:, 0:-1]
y = data[:, -1]
y = y.astype('int32')

#Ana Paola Tirado Gonzalez (Paola-Tirado)
# Nuevo punto para clasificar
nuevo_punto = [5.0, 3.3]
clase_resultado = clasificacion_knn(X, y, nuevo_punto[0], nuevo_punto[1])
print("Tipo de flor del nuevo dato:", clases[clase_resultado])


