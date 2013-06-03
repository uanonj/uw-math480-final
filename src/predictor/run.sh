#!/bin/bash

# setup the cythonized version
python decision_tree_cy_setup.py build_ext --inplace

# run the main file
python predictor.py


