import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import time

from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from Functions import gini_index, entropy
from OwnAlgorithm import DecisionTree


class GenericDecisionTree:
    def __init__(self, dataset_loader, criterion_func = gini_index, max_depth = 10, min_samples_split = 2):
        self.dataset_loader = dataset_loader
        self.criterion_func = criterion_func
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_names = None
        self.target_names = None
        
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.custom_tree = None
        self.sklearn_tree = None

    def load_data(self):
        dataset = self.dataset_loader()
        self.feature_names = dataset.feature_names
        self.target_names = dataset.target_names
        
        X, y = dataset.data, dataset.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )
    
    def train_models(self):
        print(f"Training with {len(self.X_train)} samples and {len(self.X_test)} test samples")
        self.custom_tree = DecisionTree(
            criterion_func = self.criterion_func, max_depth = self.max_depth, min_samples_split = self.min_samples_split
        )
        self.custom_tree.fit(self.X_train.tolist(), self.y_train.tolist())
        
        if self.criterion_func == entropy:
            criterion = "entropy"
        else:
            criterion = "gini"

        self.sklearn_tree = DecisionTreeClassifier(
            criterion = criterion, max_depth = self.max_depth, random_state = 42
        )
        self.sklearn_tree.fit(self.X_train, self.y_train)
    
    def evaluate_models(self):
        
        if self.custom_tree is None or self.sklearn_tree is None:
            return None
        
        y_pred_custom = self.custom_tree.predict(self.X_test.tolist())
        accuracy_custom = sum(y_pred_custom[i] == self.y_test[i] for i in range(len(self.y_test))) / len(self.y_test)
        accuracy_sklearn = self.sklearn_tree.score(self.X_test, self.y_test)
        
        print(f"Own Algorithm model accuracy: {accuracy_custom * 100:.2f}%")
        print(f"Scikit-Learn model accuracy: {accuracy_sklearn * 100:.2f}%")

        return accuracy_custom, accuracy_sklearn
    
    def run(self):
        self.load_data()
        self.train_models()
        results = self.evaluate_models()

        if results is None:
            return 0.0, 0.0
        return results


class TreeComparison:
    def __init__ (self, datasets_list, criterion_func_list, max_depth_list):
        self.datasets_list = datasets_list
        self.criterion_func_list = criterion_func_list
        self.max_depth_list = max_depth_list

        self.results = {}
        self.time_results = {}

        for criterion, _ in criterion_func_list:
            self.results[criterion] = {}
            self.time_results[criterion] = {}
            for dataset, _ in datasets_list:
                self.results[criterion][dataset] = []
                self.time_results[criterion][dataset] = []

    def compare_trees(self):
        for name, dataset_loader in self.datasets_list:
            for criterion_name, criterion_func in self.criterion_func_list:
                for max_depth in self.max_depth_list:
                    print(f"\n--- Data for Training: {name}, Criterion: {criterion_name}, Max Depth: {max_depth} ---")
                    generic_tree = GenericDecisionTree(dataset_loader=dataset_loader, criterion_func=criterion_func, max_depth=max_depth)
                    
                    start_time_custom = time.time()
                    accuracy_custom, accuracy_sklearn = generic_tree.run()
                    end_time_custom = time.time()
                    custom_time = end_time_custom - start_time_custom
                    
                    start_time_sklearn = time.time()
                    generic_tree.sklearn_tree.fit(generic_tree.X_train, generic_tree.y_train)
                    end_time_sklearn = time.time()
                    sklearn_time = end_time_sklearn - start_time_sklearn
                    
                    print(f"Adding results: {max_depth}, {accuracy_custom:.2f}%, {accuracy_sklearn:.2f}%")
                    print(f"Training Time - Custom: {custom_time:.4f} sec, Scikit-Learn: {sklearn_time:.4f} sec")
                    
                    self.results[criterion_name][name].append((max_depth, accuracy_custom * 100, accuracy_sklearn * 100))
                    self.time_results[criterion_name][name].append((max_depth, custom_time, sklearn_time))


    def plot_tree_deep_results(self):
        bar_colors = ["#1f77b4", "#084594", "#2ca02c", "#006400", "#d62728", "#800000"]  

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (criterion_name, ax) in enumerate(zip(["Gini Index", "Entropy"], axes)):
            dataset_names = list(self.results[criterion_name].keys())
            num_datasets = len(dataset_names)
            bar_width = 0.12
            spacing = 0.12

            for dataset_idx, dataset_name in enumerate(dataset_names):
                depths, custom_acc, sklearn_acc = zip(*self.results[criterion_name][dataset_name])

                x_indexes = np.arange(len(depths)) + dataset_idx * (bar_width + spacing)

                custom_color = bar_colors[(dataset_idx * 2) % len(bar_colors)]
                sklearn_color = bar_colors[(dataset_idx * 2 + 1) % len(bar_colors)]

                ax.bar(x_indexes, custom_acc, width=bar_width, label=f"{dataset_name} - Custom", alpha=0.8, color=custom_color)
                ax.bar(x_indexes + bar_width, sklearn_acc, width=bar_width, label=f"{dataset_name} - Scikit-Learn", alpha=0.8, color=sklearn_color)

            ax.set_title(f"Accuracy vs Tree Depth ({criterion_name})")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(70, 100)

            middle_x = np.arange(len(depths)) + (num_datasets * (bar_width + spacing)) / 2 - bar_width / 2
            ax.set_xticks(middle_x)
            ax.set_xticklabels(depths)

            ax.tick_params(axis="x", bottom=False)
            ax.grid(axis="y")

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=num_datasets, fontsize=10)

        plt.tight_layout()
        plt.show()


    def plot_training_time(self, custom_colors=None):
        if custom_colors is None:
            custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (criterion_name, ax) in enumerate(zip(["Gini Index", "Entropy"], axes)):
            dataset_names = [name for name in self.time_results[criterion_name].keys() if name != "Breast Cancer"]
            num_datasets = len(dataset_names)
            bar_width = 0.12
            spacing = 0.12
            
            for dataset_idx, dataset_name in enumerate(dataset_names):
                depths, custom_time, sklearn_time = zip(*self.time_results[criterion_name][dataset_name])
                x_indexes = np.arange(len(depths)) + dataset_idx * (bar_width + spacing)
                
                custom_color = custom_colors[(dataset_idx * 2) % len(custom_colors)]
                sklearn_color = custom_colors[(dataset_idx * 2 + 1) % len(custom_colors)]
                
                ax.bar(x_indexes, custom_time, width=bar_width, label=f"{dataset_name} - Custom", alpha=0.8, color=custom_color)
                ax.bar(x_indexes + bar_width, sklearn_time, width=bar_width, label=f"{dataset_name} - Scikit-Learn", alpha=0.8, color=sklearn_color)
            
            ax.set_title(f"Training Time vs Tree Depth ({criterion_name})")
            ax.set_ylabel("Training Time (s)")
            
            middle_x = np.arange(len(depths)) + (num_datasets * (bar_width + spacing)) / 2 - bar_width / 2
            ax.set_xticks(middle_x)
            ax.set_xticklabels(depths)
            
            ax.tick_params(axis="x", bottom=False)
            ax.grid(axis="y")
            ax.legend()
        
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, criterion_name in enumerate(["Gini Index", "Entropy"]):
            ax = axes[idx]
            dataset_name = "Breast Cancer"
            if dataset_name in self.time_results[criterion_name]:
                depths, custom_time, sklearn_time = zip(*self.time_results[criterion_name][dataset_name])
                x_indexes = np.arange(len(depths))
                ax.bar(x_indexes, custom_time, width=bar_width, label=f"{dataset_name} - Custom", alpha=0.8, color=custom_colors[0])
                ax.bar(x_indexes + bar_width, sklearn_time, width=bar_width, label=f"{dataset_name} - Scikit-Learn", alpha=0.8, color=custom_colors[1])
                ax.set_title(f"Training Time vs Tree Depth ({dataset_name}, {criterion_name})")
                ax.set_ylabel("Training Time (s)")
                ax.set_xticks(x_indexes)
                ax.set_xticklabels(depths)
                ax.tick_params(axis="x", bottom=False)
                ax.grid(axis="y")
                ax.legend()
        
        plt.tight_layout()
        plt.show()


datasets_list = [("Iris", load_iris), ("Wine", load_wine), ("Breast Cancer", load_breast_cancer)]
criterion_func_list = [("Gini Index", gini_index), ("Entropy", entropy)]
max_depth_list = [3, 5, 10, 20, 50]

tree_comparison = TreeComparison(datasets_list, criterion_func_list, max_depth_list)
tree_comparison.compare_trees()
tree_comparison.plot_tree_deep_results()
tree_comparison.plot_training_time()
            
