import math
import collections

UNKNOWN = "?"

cdef class DecisionTreeNode(object):
    cdef object label, attribute, value, children

    def __cinit__(self, label=None, attribute=None, value=None):
        self.label = label
        self.attribute = attribute
        self.value = value
        self.children = []

    cpdef add_child(self, child):
        if child != None:
            self.children.append(child)

    cpdef is_leaf(self):
        return len(self.children) == 0

    property label:
        def __get__(self):
            return self.label
        def __set__(self, value):
            self.label = value
        def __del__(self):
            del self.label

    property attribute:
        def __get__(self):
            return self.attribute
        def __set__(self, value):
            self.attribute = value
        def __del__(self):
            del self.attribute

    property value:
        def __get__(self):
            return self.value
        def __set__(self, value):
            self.value = value
        def __del__(self):
            del self.value

    property children:
        def __get__(self):
            return self.children
        def __set__(self, value):
            self.children = value
        def __del__(self):
            del self.children

cpdef example_target_value(example):
    """Returns the target value of the given example."""
    return example[len(example)-1]

cdef double entropy(counts, int total):
    """
    Computes the entropy of the examples represented by
    the set of counts and total.
    """
    cdef double entr, ratio
    entr = 0.0
    for value in counts:
        ratio = float(counts[value]) / total
        # small values are problematic - regard a ratio of
        # less than 0.001 as simple zero
        if ratio > 0.001:
            entr -= ratio * math.log(ratio, 2)
    return entr

cdef double information_gain(examples, attribute, values):
    """
    Computes the information gain of the examples conditioned on
    the given attribute.
    """
    cdef int total
    cdef double info_gain
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
        if ex_target_val == UNKNOWN:
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
        ratio = float(value_counts[value]) / total
        info_gain -= ratio * entropy(conditional_counts[value], value_counts[value])
    return info_gain

cdef most_common_label(examples):
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

cpdef build_tree(examples, attributes, values):
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
    result = None
    if len(examples) > 0:
        result = DecisionTreeNode()
        if len(attributes) == 0:
            # No more attributes to split on; label the node
            # and make it a root
            result.label = most_common_label(examples)
        else:
            # Determine the attribute to split on
            max_info_gain = -1
            max_attribute = 0
            for attribute in attributes:
                current_gain = information_gain(examples, attribute, values)
                if current_gain > max_info_gain:
                    max_info_gain = current_gain
                    max_attribute = attribute
            for value in values[max_attribute]:
                new_examples = []
                for example in examples:
                    if example[max_attribute] == value:
                        new_examples.append(example)
                if len(new_examples) == 0:
                    result.add_child(DecisionTreeNode(most_common_label(examples), max_attribute, value))
                else:
                    new_attributes = [a for a in attributes if a != max_attribute]
                    new_root = build_tree(new_examples, new_attributes, values)
                    new_root.attribute = max_attribute
                    new_root.value = value
                    result.add_child(new_root)
    return result

cpdef predict(root, example):
    node = root
    while not node.is_leaf():
        for child in node.children:
            if example[child.attribute] == UNKNOWN or example[child.attribute] == child.value:
                node = child
                continue
    return node.label
