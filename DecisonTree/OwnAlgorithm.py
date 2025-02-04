class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    

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


def gini_index(groups, labels):

    n_instances = float(sum([len(group) for group in groups]))

    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0

        for class_val in labels:
            proportion = group.count(class_val) / size
            score += proportion * proportion
            
        gini += (1.0 - score) * (size / n_instances)

    return gini


def get_best_split(Data, labels, criterion_func):

    best_feature, best_threshold, best_score = None, None, float("inf")
    best_groups = None
    classes = list(set(y))  # Unikalne klasy w danych

    for feature in range(len(X[0])):  # Iteracja po cechach
        thresholds = set([row[feature] for row in X])  # Unikalne wartości cechy
        for threshold in thresholds:  # Iteracja po progach
            # Podział danych na grupy
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)
            groups = [left_y, right_y]
            
            # Ocena jakości podziału
            score = criterion_func(groups, classes)
            if score < best_score:  # Szukamy minimalnej nieczystości
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


class DecisionTree:
    def __init__(self, criterion_func, max_depth=None, min_samples_split=2):
        self.criterion_func = criterion_func
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Rekurencyjnie buduje drzewo decyzyjne.

        Args:
            X (list of list): Dane wejściowe.
            y (list): Etykiety.
            depth (int): Obecna głębokość drzewa.

        Returns:
            Node: Korzeń drzewa lub poddrzewa.
        """
        num_samples, num_features = len(X), len(X[0])

        # Warunki zatrzymania
        if num_samples < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return Node(value=self._most_common_label(y))

        # Znalezienie najlepszego podziału
        split = get_best_split(X, y, self.criterion_func)
        if split["groups"] is None or len(split["groups"][0]) == 0 or len(split["groups"][1]) == 0:
            return Node(value=self._most_common_label(y))

        # Stworzenie węzłów podrzędnych
        left_X, left_y, right_X, right_y = split["groups"]
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        return Node(feature=split["feature"], threshold=split["threshold"], left=left_subtree, right=right_subtree)

    def _most_common_label(self, y):
        """Zwraca najczęściej występującą etykietę w zbiorze y."""
        return max(set(y), key=y.count)

    def predict(self, X):

        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        """
        Przechodzi przez drzewo w celu dokonania predykcji.

        Args:
            x (list): Pojedyncza próbka danych.
            node (Node): Obecny węzeł drzewa.

        Returns:
            any: Przewidziana etykieta.
        """
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)





X = [[2.3, 1.5], [1.1, 3.3], [3.3, 2.2], [1.5, 0.9], [3.0, 1.2]]
y = [0, 0, 0, 0, 0]

model = DecisionTree(criterion_func=gini_index, max_depth=3, min_samples_split=2)
model.fit(X, y)

predictions = model.predict(X)

print(predictions)  # [0, 1, 0, 1, 0]
