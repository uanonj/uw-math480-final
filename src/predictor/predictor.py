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
import decision_tree_cy
import collections
import time

CORRECT = "correct"
INCORRECT = "incorrect"

def time_prediction(f):
    start_time = time.time()
    f()
    print "time:", time.time()-start_time, "seconds"

def process(data_file):
    result = []
    for line in data_file:
        to_process = line.strip()
        if len(to_process) > 0:
            result.append(data.process_line(to_process))
    return result

def percent(num, den):
    return str(round(float(num) * 100.0 / den, 3)) + "%"

def report_results(counter):
    print CORRECT, counter[CORRECT]
    print INCORRECT, counter[INCORRECT]
    print "accuracy:", percent(counter[CORRECT], counter[CORRECT] + counter[INCORRECT])

def train_and_test(dtree_module):
    # train the decision tree
    data_file = open(data.DATA_FILE)
    examples = process(data_file)
    data_file.close()
    dtree = dtree_module.build_tree(examples, data.ATTRIBUTES, data.VALUES)
    # test the decision tree
    test_file = open(data.TEST_FILE)
    test_examples = process(test_file)
    test_file.close()
    counter = collections.Counter()
    for test in test_examples:
        prediction = dtree_module.predict(dtree, test)
        actual = dtree_module.example_target_value(test)
        # for some reason the test data appends a "." at the end
        # of each example, which messes up the predictor.
        actual = actual[0:len(actual)-1]
        if prediction == actual:
            counter[CORRECT] += 1
        else:
            counter[INCORRECT] += 1
    report_results(counter)

def train_and_test_normal():
    train_and_test(decision_tree)

def train_and_test_cython():
    train_and_test(decision_tree_cy)

def main():
    print "Beginning python version"
    time_prediction(train_and_test_normal)
    print "Done with python version"
    print "Beginning cython version"
    time_prediction(train_and_test_cython)
    print "Done."

if __name__ == '__main__':
    main()
