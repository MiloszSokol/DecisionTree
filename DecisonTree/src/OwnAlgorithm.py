import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, criterion_func, max_depth=None, min_samples_split=2):
        self.criterion_func = criterion_func
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.root = self._build_tree(X.tolist(), y.tolist(), depth=0)

    def _build_tree(self, X, y, depth):
        num_samples = len(X)

        if len(set(y)) == 1:
            return Node(value=y[0])

        if num_samples < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return Node(value=self._most_common_label(y))

        split = get_best_split(X, y, self.criterion_func)
        if split["groups"] is None or len(split["groups"][0]) == 0 or len(split["groups"][1]) == 0:
            return Node(value=self._most_common_label(y))
        
        left_X, left_y, right_X, right_y = split["groups"]
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        return Node(feature=split["feature"], threshold=split["threshold"], left=left_subtree, right=right_subtree)

    def _most_common_label(self, y):
        return max(set(y), key=y.count)

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# data = próbka np [[2.3, 1.5], [1.1, 3.3], [3.3, 2.2], [1.5, 0.9], [3.0, 1.2]]
# labels = etykiety np [0, 1, 2, 1, 0] (kwiat1, kwiat2, kwiat3)
# feature = cecha np 0, 1 (długość, szerokość)
# threshold = próg np 2.5, 1.3

# mamy tablice próbek (Data), wybieramy na podstawie której cechy(feature) chemy dzielic
# i dla kazdej próbki sprawdzamy czy wartosc cechy jest mniejsza / wieksza od progu (treashold)
# tworzymy tablice left / right od Data oraz labels (klas) i sprawdzamy czy w tablicy od labels(klas)
# zostały tylko te same klasy

#np:
#left_Data = [[2.3, 1.5], [1.1, 3.3], [1.5, 0.9]]
#left_labels = [0, 1, 1]
#right_Data = [[3.3, 2.2], [3.0, 1.2]]
#right_labels = [0, 0]

def split_dataset(Data, labels, feature, threshold):        

    right_Data, right_labels  = [], []
    left_Data, left_labels = [], []

    for i in range(len(Data)):
        if Data[i][feature] <= threshold:
            left_Data.append(Data[i])
            left_labels.append(labels[i])
        else:
            right_Data.append(Data[i])
            right_labels.append(labels[i])

    return left_Data, left_labels, right_Data, right_labels

def get_best_split(Data, labels, criterion_func):

    best_feature, best_threshold, best_score = None, None, float("inf")
    best_groups = None
    classes = list(set(labels))

    for feature in range(len(Data[0])):
        thresholds = set([row[feature] for row in Data])
        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_dataset(Data, labels, feature, threshold)
            groups = [left_y, right_y]
            
            score = criterion_func(groups, classes)
            if score < best_score: 
                best_feature = feature
                best_threshold = threshold
                best_score = score
                best_groups = (left_X, left_y, right_X, right_y)

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "groups": best_groups,
        "score": best_score
    }