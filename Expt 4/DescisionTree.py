import pandas as pd
import numpy as np


def get_probability(x: pd.Series, i) -> float:
    return x.value_counts().loc[i] / x.size


def get_entropy(x: pd.Series) -> float:

    vals = x.value_counts().index.tolist()

    entropy = 0

    for val in vals:
        prob = get_probability(x, val)
        entropy -= prob * np.log2(prob)

    return entropy


def intelligence_gain(x: pd.Series, y: pd.Series) -> float:

    entropy = get_entropy(y)

    entropies = []
    counts = []

    l = x.size

    ind = x.value_counts().index.tolist()

    for i in ind:
        x_i = x[x == i]
        y_i = y.loc[x_i.index]

        entropies.append(get_entropy(y_i))
        counts.append(x_i.size)

    ig = entropy

    for i in range(len(entropies)):
        ig -= ((counts[i]/l)*entropies[i])

    return ig


class DescisionTreeNode:

    def __init__(self, feature: str, values: list = None):

        self.feature = feature
        self.nodes = {}

        if(values):
            for val in values:
                self.nodes[val] = None

    def get_node(self, value: str):
        return self.nodes[value]

    def set_node(self, value: str, node):
        self.nodes[value] = node

    def is_leaf(self):
        return len(self.nodes) == 0

    def __str__(self):
        return self.feature+" -> "+str(self.nodes)

    def print_tree(self, tabs=0):

        for i in range(tabs):
            print("\t", end="")

        print(self.feature)
        for k, v in self.nodes.items():
            for i in range(tabs):
                print("\t", end="")
            print(f"|-> {k}:")
            v.print_tree(tabs+1)
            print()


def get_descision_tree(current_df: pd.DataFrame, target_data: pd.Series, depth: int = 0):

    cols = current_df.columns.tolist()

    max_ig = -1
    selected_col = None

    for col in cols:

        temp_ig = intelligence_gain(current_df[col], target_data)

        if temp_ig > max_ig:
            max_ig = temp_ig
            selected_col = col

    if max_ig <= 0.005 or depth >= 10:
        return DescisionTreeNode(target_data.value_counts().idxmax())

    x = current_df[selected_col]

    vals = x.value_counts().index.tolist()

    node = DescisionTreeNode(selected_col, vals)

    for val in vals:

        temp_df = current_df[current_df[selected_col] == val]
        temp_target_data = target_data.loc[temp_df.index]
        temp_node = get_descision_tree(temp_df, temp_target_data, depth+1)
        node.set_node(val, temp_node)

    return node


class DescisionTree:

    def __init__(self, training_X=None, training_y=None):
        self.training_X = training_X
        self.training_y = training_y
        self.node = None

    def fit(self, training_X, training_y):
        self.training_X = training_X
        self.training_y = training_y
        self.node = None

    def train(self):
        self.node = get_descision_tree(self.training_X, self.training_y)

    def predict(self, feature_vector):

        if(not self.node):
            raise SyntaxError("Please train the model first")

        temp = self.node

        while True:

            t = temp.get_node(feature_vector[temp.feature])

            if(t.is_leaf()):
                return t.feature

            temp = t

    def test(self, testing_X, testing_Y):
        
        if(not self.node):
            raise SyntaxError("Please train the model first")

        correct = 0
        wrong = 0

        for i in testing_X.index:
            t = self.predict(testing_X.loc[i])

            if(t == testing_Y.loc[i]):
                correct += 1
            else:
                wrong += 1

        return (100.0 * correct) / (correct + wrong)

    def view_tree(self):
        self.node.print_tree()
