import math
import time

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import timeit

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([10000,1000])
    cpu_b = tf.random.normal([1000,2000])
    print(cpu_a.device)
    gpu_a = tf.random.normal([10000,1000])
    gpu_b = tf.random.normal([1000,2000])
    print(gpu_a.device)

def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a,cpu_b)
    return c

def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a,gpu_b)
    return c

# warm up
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print("warmup", cpu_time, gpu_time)  # warmup 0.6254491 0.5488910000000002

cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print("run time", cpu_time, gpu_time)  # run time 0.5999606000000002 0.0008114999999992989

with tf.device('/gpu:0'):