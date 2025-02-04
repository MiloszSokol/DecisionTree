import unittest

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



class TestGiniIndex(unittest.TestCase):

    def test_pure_split(self):
        """Test przypadku idealnego podziału (wszystkie elementy w jednej klasie)"""
        groups = [[0, 0, 0], [1, 1, 1]]
        classes = [0, 1]
        result = gini_index(groups, classes)
        self.assertEqual(result, 0.0)  # Czysty podział, Gini = 0

    def test_max_impurity(self):
        """Test przypadku maksymalnej nieczystości (równo podzielone klasy)"""
        groups = [[0, 1], [1, 0]]
        classes = [0, 1]
        result = gini_index(groups, classes)
        self.assertEqual(result, 0.5)  # Maksymalna nieczystość, Gini = 0.5

    def test_imbalanced_split(self):
        """Test przypadku nierównego podziału (asymetria klas)"""
        groups = [[0, 0, 1], [1, 1]]
        classes = [0, 1]
        result = gini_index(groups, classes)
        expected = 0.2667  # Oczekiwana wartość (obliczona wcześniej)
        self.assertAlmostEqual(result, expected, places=2)

    def test_single_class_group(self):
        """Test przypadku, gdy jedna grupa zawiera tylko jedną klasę"""
        groups = [[0, 0, 0], []]
        classes = [0, 1]
        result = gini_index(groups, classes)
        self.assertEqual(result, 0.0)  # Brak nieczystości

    def test_empty_groups(self):
        """Test przypadku, gdy obie grupy są puste"""
        groups = [[], []]
        classes = [0, 1]
        result = gini_index(groups, classes)
        self.assertEqual(result, 0.0)  # Brak danych, Gini powinno być 0

if __name__ == "__main__":
    unittest.main()