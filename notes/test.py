import json
import pandas as pd
import numpy as np
import tensorflow as tf
a = tf.constant([[1,2,3],[4,5,6]], dtype=tf.float32)
#a= tf.ones((3,3,3))
print(tf.where(a>3).numpy() )