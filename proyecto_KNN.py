"""
Proyecto: implementación de algoritmo KNN

Autores:   Oscar Miranda Escalante A01630791 (oscarmires)
           Rodrigo Morales Aguayo A01632834 (ROmorales08)
           Ana Paola Tirado Gonzalez (Paola-Tirado)
           Roberto López Cisneros A01637335 (RobertMex)
           Marian Alejandra Herrera Ayala A00227534 (saalej)
Ultima fecha actualizacion: 29 Octubre 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mode
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA


#Roberto López Cisneros A01637335 (RobertMex)
# Grafica los puntos y los puntos proyectados
def graphPoints(X, y):  
    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X[i, 0], X[i, 1], color="red", s=10)
        elif y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color="green", s=10)
        elif y[i] == 2:
            plt.scatter(X[i, 0], X[i, 1], color="blue", s=10)
    plt.show()


# Oscar Miranda Escalante A01630791 (oscarmires)
# devuelve la clase a la que pertenece el punto
def clasificacion_knn(X, y, px, py, k=False):
    if not k:
        k = int(len(X) ** 0.5)
    nearest_neighbors = calc_distances(px, py, X).argsort()[0:k]
    mode_knn = mode(y[nearest_neighbors])
    print("Nearest neighbors' indexes:", nearest_neighbors)
    print("Moda: ", mode_knn)
    plt.scatter(px, py, color="black", marker='^', s=30)
    for i in range(k):
        plt.scatter(X[nearest_neighbors[i], 0], X[nearest_neighbors[i], 1], color="yellow", s=30)
    graphPoints(X, y)
    return mode_knn


def euclidean_distance(x0, y0, x1, y1):
    # Calcula la distancia entre dos puntos
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


# Rodrigo Morales Aguayo A01632834 (ROmorales08)
# genera un arreglo con las distancias entre un punto y todos los puntos de X
def calc_distances(x0, y0, X):
    distances = []
    for i in range(X.shape[0]):
        point = X[i, 0:2]
        distances.append(euclidean_distance(x0, y0, point[0], point[1]))
    return np.array(distances)


# Marian Alejandra Herrera Ayala A00227534 (saalej)
# importar datos
df = pd.read_csv("TC-1002-3/iris.csv")

# adecuar datos
clases = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]
df["Tipo_Flor"] = df["Tipo_Flor"].replace(clases, [0, 1, 2])
data = df.values
X = data[:, 0:-1]
y = data[:, -1]
y = y.astype('int32')

nuevo_punto = [[6, 3.1, 5.4, 2.3]]

# LDA
emb = LinearDiscriminantAnalysis(n_components=2)
emb.fit(X, y)
nuevo_punto_lda = emb.transform(np.array(nuevo_punto))
Xlda = emb.transform(X)

# PCA principal component analysis
pca = PCA(n_components=2)
pca.fit(X, y)
nuevo_punto_pca = pca.transform(np.array(nuevo_punto))
Xmds = pca.transform(X)

# Isomap
iso = Isomap(n_components=2)
iso.fit(X, y)
nuevo_punto_iso = iso.transform(np.array(nuevo_punto))
Xiso = iso.fit_transform(X)

#Ana Paola Tirado Gonzalez (Paola-Tirado)
# nuevo punto para clasificar
# clasificacion_knn(todos los puntos, clases, coordx nuevo punto, coordy nuevo punto)
clase_resultado_lda = clasificacion_knn(Xlda, y, nuevo_punto_lda[0][0], nuevo_punto_lda[0][1], k=4)
clase_resultado_pca = clasificacion_knn(Xmds, y, nuevo_punto_pca[0][0], nuevo_punto_pca[0][1], k=4)
clase_resultado_iso = clasificacion_knn(Xiso, y, nuevo_punto_iso[0][0], nuevo_punto_iso[0][1], k=4)
print("Tipo de flor del nuevo dato (LDA):", clases[clase_resultado_lda])
print("Tipo de flor del nuevo dato (PCA):", clases[clase_resultado_pca])
print("Tipo de flor del nuevo dato (Isomap):", clases[clase_resultado_iso])
