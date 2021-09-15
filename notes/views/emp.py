from flask import Blueprint, jsonify
from util import mysql_util
import json
bp = Blueprint('emp', __name__)


@bp.route('/find/<int:id>', methods=['GET', 'POST'])
def employee(id):
    bd = mysql_util.BaseDao('company')
    data = bd.find_all('t_emp', 'where e_id=%d;' % id)
    temp = [(k, v) for d in data for k, v in d.items()]
    res = ''
    for i in temp:
        res += str(i[0])+':'+str(i[1])+','
    return res
