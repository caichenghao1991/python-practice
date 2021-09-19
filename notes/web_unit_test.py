from unittest import TestCase

import requests
import unittest
from flask_notes import *
from models import db
from models.dept import t_dept

class TestEmp(unittest.TestCase):

    def test_sum(self):
        url = "http://localhost:5000/emp/find/1"
        method = 'get'
        resp = requests.request(method, url)
        self.assertEqual(resp.status_code, 200, 'Request failed')
        print(resp.text)




if __name__ == '__main__':
    unittest.main()
