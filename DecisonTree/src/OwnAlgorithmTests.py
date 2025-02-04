from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from Functions import gini_index, entropy
from sklearn.model_selection import train_test_split
from OwnAlgorithm import DecisionTree
from Functions import gini_index, entropy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def print_tree(node, depth=0):
    if node.is_leaf():
        print(f"{'  ' * depth}--> Class: {node.value}")
    else:
        print(f"{'  ' * depth}[Feature {node.feature} <= {node.threshold}]")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

iris = load_iris()
X, y = iris.data, iris.target

test_sizes = np.arange(0.1, 0.6, 0.05)

train_accuracies = []
test_accuracies = []


for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    tree = DecisionTree(criterion_func=gini_index, max_depth=10)
    tree.fit(X_train.tolist(), y_train.tolist())

    y_train_pred = tree.predict(X_train.tolist())
    y_test_pred = tree.predict(X_test.tolist())

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)


print_tree(tree.root)

plt.figure(figsize=(8, 5))
plt.plot(test_sizes, train_accuracies, label="Training Accuracy", marker="o")
plt.plot(test_sizes, test_accuracies, label="Testing Accuracy", marker="s")
plt.xlabel("Test Data Ratio")
plt.ylabel("Accuracy")
plt.title("Effect of Train-Test Split on Model Performance")
plt.legend()
plt.grid()
plt.show()
