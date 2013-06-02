import math
import collections

UNKNOWN = "?"

class DecisionTreeNode(object):
    def __init__(self, label=None):
        self.label = label
        self.children = []

    def add_child(self, child):
        if child != None:
            self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

def example_target_value(example):
    """Returns the target value of the given example."""
    return example[len(example)]

def entropy(examples):
    """Calculates the entropy of the given examples."""
    total = len(examples)
    result = 0.0
    counts = collections.Counter()
    for example in examples:
        counts[example_target_value(example)] += 1
    for target_value in counts:
        ratio = float(counts[target_value]) / total
        # small values are problematic - regard a ratio of
        # less than 0.001 as simple zero
        if ratio > 0.001:
            result -= ratio * math.log(ratio, 2)
    return result

def entropy(counts, total):
    """
    Computes the entropy of the examples represented by
    the set of counts and total.
    """
    entr = 0.0
    for value in counts:
        ratio = float(counts[value]) / total
        # small values are problematic - regard a ratio of
        # less than 0.001 as simple zero
        if ratio > 0.001:
            entr -= ratio * math.log(ratio, 2)
    return entr

def information_gain(examples, attribute, values):
    """
    Computes the information gain of the examples conditioned on
    the given attribute.
    """
    total = len(examples)
    counts = collections.Counter()
    value_counts = collections.Counter()
    # conditional_counter: attribute values -> counter for target values
    # partitions the example set to calculate the data
    conditional_counts = {}
    for value in values[attribute]:
        conditional_counts[value] = collections.Counter()
    for example in examples:
        ex_target_val = example_target_value(example)
        if ex_target_val == UKNOWN:
            continue
        ex_attr_val = example[attribute]
        counts[ex_target_val] += 1
        if ex_attr_val != UNKNOWN:
            conditional_counts[ex_attr_val][ex_target_val] += 1
            value_counts[ex_attr_val] += 1
    # calculate the information gain
    # initialize to the entropy of the example set
    info_gain = entropy(counts, total)
    # now take into account the entropy conditioned on the attribute
    for value in values[attribute]:
        ratio = float(conditional_counts[value]) / total
        info_gain -= ratio * entropy(conditional_counts[value], value_counts[value])
    return info_gain

def most_common_label(examples):
    """Returns the most common target label among the examples"""
    count = collections.Counter()
    for example in examples:
        count[example_target_value(example)] += 1
    max_label_count = -1
    max_label = None
    for label in count:
        if count[label] > max_label_count:
            max_label_count = count[label]
            max_label = label
    return max_label

def build_tree(examples, attributes, values):
    """
    build_tree Recursively builds a decision tree using information gain as the splitting criterion.

    Parameters:
        - examples
            the remaining examples to be classified
        - attributes
            the remaining attributes to be considered by the tree
        - values
            a dict: attribute->values that maps to possible values of an attribute
    Returns:
        A DecisionTreeNode that represents the decision tree under the given
        examples and attributes.
    """
    result = DecisionTreeNode()
    if len(examples) > 0:
        if len(attributes) == 0:
            result.label = most_common_label(examples)
        else:
            max_info_gain = -1
            max_attribute = 0
            for attribute in attributes:
                current_gain = information_gain(examples, attribute, value)
                if current_gain > max_info_gain:
                    max_info_gain = current_gain
                    max_attribute = attribute
            for value in values[max_attribute]:
                new_examples = []
                for example in examples:
                    if example[max_attribute] == value:
                        new_examples.append(example)
                if len(new_examples) == 0:
                    result.addChild(DecisionTreeNode(most_common_label(examples)))
                else:
                    new_attributes = [a for a in attributes if a != max_attribute]
                    result.addChild(build_tree(new_examples, new_attributes, values))
    return result
