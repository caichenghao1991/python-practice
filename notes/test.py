from multiprocessing import Process

import os

import json
import pandas as pd
import numpy as np
import random
from numpy import mean
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

b2 = tf.tile(tf.ones((20,1,3)),[2,1,1])
print(b2.shape)

