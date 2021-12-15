from multiprocessing import Process

import os

import json
import pandas as pd
import numpy as np
import random
def func(v):
    n = 0
    while n<v:
        print(n)
        x = yield n   # pause the function until next call of function
                    # temp = yield n   # receive input parameter from outside
        print(x)
        n += 1

g = func(7)
g.send(None)
g.send(5)
g.send(5)
g.send(5)
next(g)