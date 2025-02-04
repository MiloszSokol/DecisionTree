import unittest

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


def gini_index(groups, classes):

    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0

    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0

        for class_val in classes:
            proportion = group.count(class_val) / size
            score += proportion * proportion
            
        gini += (1.0 - score) * (size / n_instances)

    return gini


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

            print(f"DEBUG: Cecha {feature}, Próg {threshold}, Gini {score}")
            
            if score < best_score:
                best_feature = feature
                best_threshold = threshold
                best_score = score
                best_groups = (left_X, left_y, right_X, right_y)

    print(f"Najlepszy podział: cecha {best_feature}, próg {best_threshold}, Gini {best_score}")

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "groups": best_groups,
        "score": best_score
    }


class TestGetBestSplit(unittest.TestCase):

    def test_correct_best_split(self):
        """Test sprawdzający, czy funkcja znajduje najlepszy podział"""
        X = [[2.3, 1.5], [1.1, 3.3], [3.3, 2.2], [1.5, 0.9], [3.0, 1.2]]
        y = [0, 1, 0, 1, 0]

        result = get_best_split(X, y, gini_index)

        #print("DEBUG - Wynik funkcji get_best_split:")
        #print(result)

        expected_feature = 0  # Podział według pierwszej cechy (długość)
        expected_threshold = 1.5  # Najlepszy próg podziału
        expected_score = 0.0  # Przykładowa wartość Giniego
        
        self.assertEqual(result["feature"], expected_feature)
        self.assertEqual(result["threshold"], expected_threshold)
        self.assertAlmostEqual(result["score"], expected_score, places=2)

    def test_split_on_different_feature(self):
        """Test, czy funkcja może podzielić według drugiej cechy (szerokość)"""
        X = [[2.3, 1.5], [1.1, 3.3], [3.3, 2.2], [1.5, 0.9], [3.0, 1.2]]
        y = [0, 1, 0, 1, 0]
        result = get_best_split(X, y, gini_index)

        self.assertIn(result["feature"], [0, 1])  # Powinna wybrać którąś z cech
        self.assertIn(result["threshold"], [1.5, 2.3, 3.3])  # Możliwe wartości progów

if __name__ == "__main__":
    unittest.main()
