import collections
import hashlib
import heapq
import sys
import urllib
import datetime
from multiprocessing import Process

import os

import json
import pandas as pd
import numpy as np
import random

import pymysql
import requests
from numpy import mean
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier


class Node():
    def __init__(self, value=None):
        self.value = value
        self.next = None


class LinkedList():
    def __init__(self, value):
        self.head = Node(value)  # head is the first element of the linked list
        self.length = 1
        self.tail = self.head  # tail is the end of linked list

def pr(node):
    while node:
        print(node.value)
        node=node.next


def reverse(head):
    pre = Node()
    while head:
        nex = head.next
        head.next = pre
        pre = head
        head = nex
    return pre
l = LinkedList(1)
l.head.next = Node(2)
l.head.next.next = Node(3)

df=pd.DataFrame({'a':[1,2,3],'b':[4,5,6]},index=[1,2,3])
print(df[['a','b']])
print(datetime.datetime.now().year)
plt.figure()
ax=plt.subplot(1,2,1)
plt.hist([1,2,3,4,5,6,7,8,9,10,1,2,3,4,1,2], bins=10)
ax2=plt.subplot(1,2,2)
plt.hist([1,2,3,4,5,6,7,8,9,10,1,2,3,4,1,2], bins=10)
plt.show()