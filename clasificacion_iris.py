from sklearn import tree
import pandas as pd
import numpy as np

iris1 = pd.read_csv("iris1.csv")
iris2 = pd.read_csv("iris2.csv")

iris1["Tipo_Flor"] = iris1["Tipo_Flor"].replace(["Iris-versicolor", "Iris-virginica", "Iris-setosa"], [0, 1, 2])
data = iris1.values
X = data[:, 0:-1]
y = data[:, -1]

X2 = iris2.values

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
y2hat = clf.predict(X2)
