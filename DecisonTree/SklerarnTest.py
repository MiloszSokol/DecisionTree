from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()

#DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Mapowanie etykiet
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(data.head())

# Podział danych
X = data[iris.feature_names]  # cechy
y = data['species']           # etykiety

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tworzenie modelu
model = DecisionTreeClassifier(random_state=42)

# Trenowanie modelu na danych treningowych
model.fit(X_train, y_train)

print("Model został wytrenowany.")

# Predykcje
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Ocena dokładności
print("Dokładność na treningowym:", accuracy_score(y_train, y_pred_train))
print("Dokładność na testowym:", accuracy_score(y_test, y_pred_test))

# Rysowanie drzewa decyzyjnego
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()