uw-math480-final
================

UW Math 480, Spring 2013, Final Project  
Jason Uanon, Ayla Lampard  
Last Updated: May 14, 2013  

Our project will attempt to predict whether a person's income exceeds $50k/yr in 1994 based on census data for that year. We will use the "Adult Data Set" from the UCI Machine Learning Repository[1], which contains over 32,000 training instances and over 16,000 test instances of census data containing 14 attributes (http://archive.ics.uci.edu/ml/datasets/Adult).

Using this data, throwing out data with missing values if needed, we will predict whether a person's income exceeds $50k/yr using the methods of information theory to build a decision tree out of a set of training data, and will predict the incomes of the remaining data.

We will be using Python 2.7.3 in conjunction with Cython 0.15.1 for performance enhancements and the Sage Math Cloud. Part of our project will be writing the initial version in pure Python and comparing the performance with Python + Cython/Sage.


[1] Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
