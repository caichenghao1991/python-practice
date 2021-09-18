from flask import Blueprint, make_response, Response
from werkzeug.exceptions import abort

from util import mysql_util
from flask import url_for, redirect, request
import json
from typing import List

bp = Blueprint('emp', __name__)


@bp.route('/show/<data>')
def display(data):
    return "Your information is:<br>%s" % data


@bp.route('/find/<int:id>', methods=['GET', 'POST'])  # <int:id>
def employee(id):
    bd = mysql_util.BaseDao('company')
    data = bd.find_all('t_emp', 'where e_id=%d;' % id)

    if not data:
        #abort(403)  # status code. return error page with status code
        abort(Response("data not valid", 403))  # response page with error information

    temp = [(k, v) for d in data for k, v in d.items()]
    res = ''
    for i in temp:
        res += str(i[0]) + ':' + str(i[1]) + ','

    # return redirect("%s" % url_for('emp.display', data=res))  #  , code=200  don't need specific display logic

    return """
            Found your information, <a href='%s'>showing your query result in 3 second</a>
            <script>url=document.querySelector('a').href; setInterval(function(){window.location.href=url;}, 3000)</script>
            """ % url_for('emp.display', data=res)

    """
        response = make_response(data, 200)   # data need to be json (careful byte and date need to convert)
        response.headers['Content-Type'] = 'application/json;charset=utf-8'
        # response = Response(data, 200, content-type='application/json')
        return response
    """
