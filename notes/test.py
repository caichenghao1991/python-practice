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


def task1(n):
    for i in range(n):
        print(n)
        yield None
birthday = datetime.datetime(2019,6,20,10,30)
print(datetime.datetime.ctime(birthday))


