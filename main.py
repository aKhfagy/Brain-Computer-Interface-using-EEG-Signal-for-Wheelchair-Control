# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:07:13 2021

@author: Ahmed
"""

from FNN import FNN
from datasets import TUAR2

features, labels, n_output = TUAR2()

fuzzy = FNN(features, labels)
fuzzy.make_model(n_inputs=4, n_hidden=4, n_outputs=n_output, n_iterations=100)

