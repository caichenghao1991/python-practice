import json
import pandas as pd
import numpy as np
df = pd.DataFrame({'a':[0, 1],'b':[2, 6],'c':[0,1]}, index=['x','y'])
delta = pd.Series([10, -2], index = ['x','y'])
df = df.add(delta, axis='index')

print(df.corr())


def out(func):

    def inn(x):
        print(x)

        return x

    return inn

@out
def a(y):
   return y

xx =a(2)
print(xx)