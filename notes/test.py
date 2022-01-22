import collections
import hashlib
import heapq
import sys
import urllib
import datetime
from multiprocessing import Process
import redis
import os
import tensorflow as tf

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


def task1(n):
    for i in range(n):
        print(n)
        yield None


def strStr(haystack, needle):  # needle is pattern
    n, h = len(needle), len(haystack)
    i, j, lps = 1, 0, [-1] + [0] * n  # i: pointer for pattern, j: pointer for string, lps array
    while i < n:  # calculate next array
        if j == -1 or needle[i] == needle[j]:
            i += 1
            j += 1
            lps[i] = j
        else:
            j = lps[j]
    i = j = 0
    print(lps)
    while i < h and j < n:
        if j == -1 or haystack[i] == needle[j]:
            i += 1
            j += 1
        else:
            j = lps[j]
    return i - j if j == n else -1


