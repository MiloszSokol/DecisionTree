from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd

def display_dataset_info(dataset, name):
    print(f"\n--- {name} Dataset ---")
    print("Feature:", dataset.feature_names)
    print("Class:", dataset.target_names)
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    print( "\n", df.head())
    print( "---------------------------------------------------------------------------------------------------------")

iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()

display_dataset_info(iris, "Iris")
display_dataset_info(wine, "Wine")
display_dataset_info(cancer, "Breast Cancer")