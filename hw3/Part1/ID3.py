import numpy as np


# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # print(self.calculate_average_entropy__(self.dataset, self.labels, "Outlook")) # should be -> 0.693

        # print(self.calculate_information_gain__(self.dataset, self.labels, "Humidity"))  # should be -> 0.151
        # print(self.calculate_information_gain__(self.dataset, self.labels, "Windy"))  # should be -> 0.048
        # print(self.calculate_information_gain__(self.dataset, self.labels, "Temperature"))  # should be -> 0.029

        # print(self.calculate_intrinsic_information__(self.dataset, self.labels, "Outlook"))  # should be -> 1.577
        # print(self.calculate_intrinsic_information__(self.dataset, self.labels, "Temperature"))  # should be -> 1.557
        # print(self.calculate_intrinsic_information__(self.dataset, self.labels, "Humidity"))  # should be -> 1.000
        # print(self.calculate_intrinsic_information__(self.dataset, self.labels, "Windy"))  # should be -> 0.985

        # print(self.calculate_gain_ratio__(self.dataset, self.labels, "Outlook"))  # should be -> 0.157
        # print(self.calculate_gain_ratio__(self.dataset, self.labels, "Temperature"))  # should be -> 0.019
        # print(self.calculate_gain_ratio__(self.dataset, self.labels, "Humidity"))  # should be -> 0.152
        # print(self.calculate_gain_ratio__(self.dataset, self.labels, "Windy"))  # should be -> 0.049

    def calculate_entropy__(self, dataset, labels):
        # ACCORDING TO LECTURE MATERIAL
        # DecTrees.pdf page 12
        entropy_value = 0.0
        uniq, counts = np.unique(labels, return_counts=True)
        for i in range(len(uniq)):
            p = counts[i] / len(labels)
            entropy_value += -p * np.log2(p)

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        # SLIDES PAGE 14
        average_entropy = 0.0
        index = self.features.index(attribute)
        col = np.array(dataset)[:, index]
        keys_and_yes = {}
        for i in range(len(col)):
            if col[i] not in keys_and_yes:
                keys_and_yes[col[i]] = [labels[i]]
            else:
                keys_and_yes[col[i]].append(labels[i])

        for key in keys_and_yes:
            entropy = self.calculate_entropy__(dataset, keys_and_yes[key])
            average_entropy += (len(keys_and_yes[key]) / len(labels)) * entropy

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        # SLIDES PAGE 15
        entropy_of_set = self.calculate_entropy__(dataset, labels)
        avg_entropy = self.calculate_average_entropy__(dataset, labels, attribute)
        return float(entropy_of_set - avg_entropy)

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        # SLIDES PAGE 24
        intrinsic_info = 0.0
        index = self.features.index(attribute)
        col = np.array(dataset)[:, index]
        uniq, counts = np.unique(col, return_counts=True)
        for i in range(len(uniq)):
            intrinsic_info += (-counts[i] / len(col)) * np.log2(counts[i] / len(col))
        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        gain_ratio = self.calculate_information_gain__(
            dataset, labels, attribute
        ) / self.calculate_intrinsic_information__(dataset, labels, attribute)
        return gain_ratio

    def ID3__(self, dataset, labels, used_attributes):
        uniq = np.unique(labels)
        if len(uniq) <= 1 or len(used_attributes) == len(self.features):
            uniq, counts = np.unique(labels, return_counts=True)
            majority_label = uniq[np.argmax(counts)]
            return TreeLeafNode(dataset, majority_label)

        best_attribute = None
        best_gain = 0

        # FIND THE SPLIT ATTRIBUTE

        gain = 0
        for i in range(len(self.features)):
            if self.features[i] in used_attributes:
                continue
            if self.criterion == "information gain":
                gain = self.calculate_information_gain__(
                    dataset, labels, self.features[i]
                )
            elif self.criterion == "gain ratio":
                gain = self.calculate_gain_ratio__(dataset, labels, self.features[i])
            if gain > best_gain:
                best_gain = gain
                best_attribute = self.features[i]

        node = TreeNode(best_attribute)

        used_attributes.append(best_attribute)
        index = self.features.index(best_attribute)
        col = np.array(dataset)[:, index]
        uniq = np.unique(col)

        # CALL ID3 RECURSIVELY FOR EACH VALUE OF THE SPLIT ATTRIBUTE

        for i in range(len(uniq)):
            new_dataset = []
            new_labels = []
            for j in range(len(dataset)):
                if dataset[j][index] == uniq[i]:
                    new_dataset.append(dataset[j])
                    new_labels.append(labels[j])
            node.subtrees[uniq[i]] = self.ID3__(
                new_dataset, new_labels, used_attributes.copy()
            )

        return node

    def predict(self, x):
        current_node = self.root

        while isinstance(current_node, TreeNode):
            index = self.features.index(current_node.attribute)
            attribute_value = x[index]
            if attribute_value in current_node.subtrees:
                current_node = current_node.subtrees[attribute_value]
            else:
                break

        if isinstance(current_node, TreeLeafNode):
            uniq, counts = np.unique(current_node.labels, return_counts=True)
            majority_label = uniq[np.argmax(counts)]
            return majority_label

        return None

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")
