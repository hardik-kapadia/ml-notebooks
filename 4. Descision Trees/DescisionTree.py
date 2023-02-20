from random import randint
from typing import List
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

    # entropy = get_entropy(y)
    ig = get_entropy(y)

    # entropies = []
    # counts = []

    l = x.size

    ind = x.value_counts().index.tolist()

    for i in ind:
        x_i = x[x == i]
        y_i = y.loc[x_i.index]

        ig -= (x_i.size / l) * get_entropy(y_i)

        # entropies.append(get_entropy(y_i))
        # counts.append(x_i.size)

    # ig = entropy

    # for i in range(len(entropies)):
    #     ig -= ((counts[i]/l)*entropies[i])

    return ig


def print_tree(node, tabs=0):
    for i in range(tabs):
        print("|\t", end="")
    print(node.feature)
    for k, v in node.nodes.items():
        for _ in range(tabs):
            print("|\t", end="")
        print(f"|-> {k}:")
        v.print_tree(tabs + 1)
        
class DescisionTreeNode:
    def __init__(self, feature: str, values: List = None):  # type: ignore

        self.feature = feature
        self.nodes = {}

        if values:
            for val in values:
                self.nodes[val] = None

    def get_node(self, value: str):
        return self.nodes.get(value)

    def set_node(self, value: str, node):
        self.nodes[value] = node

    def is_leaf(self):
        return len(self.nodes) == 0

    def __str__(self):
        return self.feature + " -> " + str(self.nodes)

def get_descision_tree(
    current_df: pd.DataFrame,
    target_data: pd.Series,
    depth: int = 0,
    max_depth: int = 10,
    ig_limit: float = 0.001,
):
    cols = current_df.columns.tolist()

    max_ig = -1
    selected_col: str = ""

    for col in cols:

        temp_ig = intelligence_gain(current_df[col], target_data)

        if temp_ig > max_ig:
            max_ig = temp_ig
            selected_col = col

    if max_ig <= ig_limit or depth >= max_depth:
        return DescisionTreeNode(str(target_data.value_counts().idxmax()))

    vals = current_df[selected_col].value_counts().index.tolist()

    node = DescisionTreeNode(selected_col, vals)

    for val in vals:

        temp_df = current_df[current_df[selected_col] == val]
        temp_target_data = target_data.loc[temp_df.index]
        temp_df_2 = temp_df.drop(selected_col, axis=1)
        temp_node = get_descision_tree(temp_df_2, temp_target_data, depth + 1)
        node.set_node(val, temp_node)

    return node


class DescisionTree:
    def __init__(self, training_X: pd.DataFrame, training_y: pd.Series):
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

        if not self.node:
            raise SyntaxError("Please train the model first")

        temp = self.node

        while True:

            t = temp.get_node(feature_vector[temp.feature])

            if t.is_leaf():
                return t.feature

            temp = t

    def test(self, testing_X, testing_Y):

        if not self.node:
            raise SyntaxError("Please train the model first")

        correct = 0
        wrong = 0

        for i in testing_X.index:
            t = self.predict(testing_X.loc[i])

            if t == testing_Y.loc[i]:
                correct += 1
            else:
                wrong += 1

        return (100.0 * correct) / (correct + wrong)

    def view_tree(self):
        print_tree(self.node)


class BaggedDecisionTree:
    def __init__(
        self, training_X: pd.DataFrame, training_y: pd.Series, trees: int = 32
    ):
        self.training_X = training_X
        self.training_y = training_y
        self.trees = trees
        self.nodes = []

    def fit(self, training_X, training_y):
        self.training_X = training_X
        self.training_y = training_y
        self.nodes = []

    def get_sampled_data(self):

        n = self.training_y.size

        choices = [randint(0, n * 3) % n for _ in range(n)]

        sampled_X = self.training_X.iloc[choices]
        sampled_Y = self.training_y.iloc[choices]

        sampled_X.reset_index(drop=True, inplace=True)
        sampled_Y.reset_index(drop=True, inplace=True)

        return (sampled_X, sampled_Y)

    def train(self):

        for _ in range(self.trees):

            temp_train_X, temp_train_Y = self.get_sampled_data()

            self.nodes.append(get_descision_tree(temp_train_X, temp_train_Y))

    def predict(self, feature_vector):

        if len(self.nodes) == 0:
            raise SyntaxError("Please train the model first")

        votes = {}

        for node in self.nodes:

            vote = self.predict_single_tree(feature_vector, node)
            if vote:
                val = votes.get(vote, 0) + 1
                votes[vote] = val

        n = len(self.nodes)

        max_votes = -1
        highest = None

        for k, v in votes.items():

            if v > max_votes:
                highest = k
                max_votes = v

        return (highest, votes.get(highest) / n)

    def predict_single_tree(self, feature_vector, node):

        temp = node

        while True:

            t = temp.get_node(feature_vector[temp.feature])

            if t == None:
                return None

            if t.is_leaf():
                return t.feature

            temp = t

    def test(self, testing_X, testing_Y):

        if len(self.nodes) == 0:
            raise SyntaxError("Please train the model first")

        correct = 0
        wrong = 0

        for i in testing_X.index:
            t = self.predict(testing_X.loc[i])

            if t[0] == testing_Y.loc[i]:
                correct += 1
            else:
                wrong += 1

        return (100.0 * correct) / (correct + wrong)
