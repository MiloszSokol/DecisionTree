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


class TestSplitDataset(unittest.TestCase):
    
    def test_split_correctly(self):
        X = [[2.3, 1.5], [1.1, 3.3], [3.3, 2.2], [1.5, 0.9], [3.0, 1.2]]
        y = [0, 1, 0, 1, 0]
        feature = 0  # Cecha: pierwsza kolumna (długość)
        threshold = 2.5  # Próg

        left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)

        expected_left_X = [[2.3, 1.5], [1.1, 3.3], [1.5, 0.9]]
        expected_left_y = [0, 1, 1]
        expected_right_X = [[3.3, 2.2], [3.0, 1.2]]
        expected_right_y = [0, 0]

        self.assertEqual(left_X, expected_left_X)
        self.assertEqual(left_y, expected_left_y)
        self.assertEqual(right_X, expected_right_X)
        self.assertEqual(right_y, expected_right_y)

        print(left_X, left_y, right_X, right_y)

if __name__ == "__main__":
    unittest.main()