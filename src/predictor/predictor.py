"""
predictor.py
Given a module data with arbitrary attributes, predictor 
generates a decision tree from the data file and uses tests
the tree on the test data.

data must satisfy the following interface:
    NUM_ATTRIBUTES
        a constant indicating the number of attributes
    initialize_totals()
    initialize_attribute_counts()
    classify(attr, value, target, totals, attribute_counts)
"""

import data
import decision_tree

def process(data_file, total_data, attribute_counts):
    for line in data_file:
        if len(line) > 0:
            line_data = line.split(", ")
            # stop the loop one short; the last attribute
            # is the target attribute
            for i in range(len(line_data) - 1):
                data.classify(i, line_data[i], line_data[data.NUM_ATTRIBUTES].strip(), total_data, attribute_counts)

def main():
    # train the decision tree
    data_file = open(data.DATA_FILE)
    total_data = data.initialize_totals()
    attribute_counts = data.initialize_attribute_counts()
    process(data_file, total_data, attribute_counts)
    data_file.close()
    # test the decision tree
    test_file = open(data.TEST_FILE)

if __name__ == '__main__':
    main()
