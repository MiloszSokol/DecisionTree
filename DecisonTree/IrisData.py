from sklearn.datasets import load_iris
import pandas as pd

# Załadowanie danych Iris
iris = load_iris()

print(iris.feature_names)  # Nazwy cech
print(iris.target_names)   # Nazwy klas

print(iris.data)    # Wartości cech
print(iris.target)   # Wartości klas
