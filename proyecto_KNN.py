"""
Proyecto: implementación de algoritmo KNN

Autores:   Oscar Miranda Escalante A01630791 (oscarmires)
           Rodrigo Morales Aguayo A01632834 (ROmorales08)
           Ana Paola Tirado Gonzalez (Paola-Tirado)
           Roberto López Cisneros A01637335 (RobertMex)
           Marian Alejandra Herrera Ayala A00227534 (saalej)
           José Miguel Figarola Prado A01632557 (josefigarola)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mode
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

#Roberto López Cisneros A01637335 (RobertMex)
# Grafica los puntos y los puntos proyectados
def graphPoints(X, y):  
    for i in range(X.shape[0]):
        p = X[i, 0:2]
        if y[i] == 0:
            plt.scatter(p[0], p[1], color="red", s=4)
        elif y[i] == 1:
            plt.scatter(p[0], p[1], color="green", s=4)
        elif y[i] == 2:
            plt.scatter(p[0], p[1], color="blue", s=4)
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

# José Miguel Figarola Prado A01632557 (josefigarola)
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

nuevo_punto = [[6.2, 3.3, 4, 1.4]]

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
nuevo_punto_lda = lda.transform(nuevo_punto)
Xlda = lda.transform(X)

pca = PCA(n_components=2)
pca.fit(X, y)
nuevo_punto_pca = pca.transform(nuevo_punto)
Xpca = pca.transform(X)

iso = Isomap(n_components=2)
iso.fit(X, y)
nuevo_punto_iso = iso.transform(nuevo_punto)
Xiso = iso.transform(X)

#Ana Paola Tirado Gonzalez (Paola-Tirado)
# nuevo punto para clasificar
# clasificación recibe argumentos: todos los puntos (X), todas las clases (y), coordx nuevo punto, coordy nuevo punto
clase_resultado_lda = clasificacion_knn(Xlda, y, nuevo_punto_lda[0][0], nuevo_punto_lda[0][1])
clase_resultado_pca = clasificacion_knn(Xpca, y, nuevo_punto_pca[0][0], nuevo_punto_pca[0][1])
clase_resultado_iso = clasificacion_knn(Xiso, y, nuevo_punto_iso[0][0], nuevo_punto_iso[0][1])
print("Tipo de flor del nuevo dato (LDA):", clases[clase_resultado_lda])
print("Tipo de flor del nuevo dato (PCA):", clases[clase_resultado_pca])
print("Tipo de flor del nuevo dato (Isomap):", clases[clase_resultado_iso])
