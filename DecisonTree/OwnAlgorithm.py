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
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = len(X), len(X[0])

        if num_samples < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return Node(value=self._most_common_label(y))

        best_feature, best_threshold = None, None
        best_score = float("-inf")
        for feature in range(num_features):
            thresholds = set(row[feature] for row in X)
            for threshold in thresholds:
                score = self.criterion_func(X, y, feature, threshold)
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return Node(value=self._most_common_label(y))

        left_indices = [i for i in range(num_samples) if X[i][best_feature] <= best_threshold]
        right_indices = [i for i in range(num_samples) if X[i][best_feature] > best_threshold]

        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]

        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

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

def gini_index(X, y, feature, threshold):
    left_y = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
    right_y = [y[i] for i in range(len(X)) if X[i][feature] > threshold]

    def gini(group):
        size = len(group)
        if size == 0:
            return 0
        proportions = [group.count(label) / size for label in set(group)]
        return 1 - sum(p ** 2 for p in proportions)

    n = len(y)
    gini_left = gini(left_y)
    gini_right = gini(right_y)
    weighted_gini = (len(left_y) / n) * gini_left + (len(right_y) / n) * gini_right
    return -weighted_gini

X = [[2.3, 1.5], [1.1, 3.3], [3.3, 2.2], [1.5, 0.9], [3.0, 1.2]]
y = [0, 1, 0, 1, 0]

model = DecisionTree(criterion_func=gini_index, max_depth=3, min_samples_split=2)
model.fit(X, y)

print("Predictions:", model.predict(X))
