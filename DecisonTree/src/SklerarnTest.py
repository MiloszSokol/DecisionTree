import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def train_and_plot_tree(dataset, dataset_name):
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names
    class_names = dataset.target_names.astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth= 10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"{dataset_name} - Training Accuracy: {train_acc:.2f}, Testing Accuracy: {test_acc:.2f}")
    
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title(f"Decision Tree for {dataset_name}")
    plt.show()

datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Breast Cancer": load_breast_cancer()
}

for name, dataset in datasets.items():
    train_and_plot_tree(dataset, name)