"""
    Web application service is based on HTTP protocol, handling request for static resource and dynamic resource.
    static resource: webpage, image, css, js script, video, audio.
    dynamic resource: created by backend program (java/php/python/.net) and database (mysql/oracle/sqlserver).
        generating html response to client base on business requirement

    client (browser) packaging a Http Request, send to server. Server receive request and analyze request path. Server
        based on request path, read / generate byte data and packaging into Http Response. Client receive Http
        Response,if request successful, shows the http respond data (html/json)

    Python provide wsgiref module (basic, core) to implement web application interface. Flask used Werkzeug module
    implement WSGI protocol.
    WSGI: (web server gateway interface), responsible for communication under http protocol

    Python web framework: Django(most modules, used for server management, operation and maintenance(Ansible/Openstack))
        Flask(small, flexible, used for API); Tornado based on coroutines, single thread,single process asynchronous;
        Sanic: good performance asynchronous framework;




    make_response('%s OK' % status_code, header)  # native python implementing wsgi,
        # header:[('content-type', 'images\*')]   'text/html;charset=utf-8'  'audio/mpeg'   'video/mp4'
    return body  # content inside body

    Flask use M (model) T (templates, html) V (views, controller)  which is based on MVC.
    templates: html file with {{var}}   var is send in the response via
    flask request object contains dict attribute
    request.args: query parameter, url second part separated by ?
    request.form: form parameter, mostly data in the post request
    request.headers: status, browser/system information
    request.cookies   Cookie data
    request.files    file uploaded
    request.method:  upper case request method
    request.base_url: url not including get parameter (after ?),  host_url: only ip and port,  path: routing path /find
        remote_addr: client ip

    both request and response have headers and bodies
    response = requests.request(method, url)   # requests are used for testing, simulate browser send request
    response is the return object in the views functions
    from flask import render_template   (based on Jinja2 package)
    app = Flask(__name__, template_folder=template_folder, static_folder='resource')  # set templates folder location, if
        not under same parent and folder name is templates. static default folder is same parent, and named static
        static_url_path='static'  default is /    # add request prefix for static folder item
        <link rel="stylesheet" href="/static/css/my.css">  to access the css file in resource/css folder
        same as set in config file: STATIC_FOLDER = '/resource'   STATIC_URL_PARH = '/static'
        or app.static_url_path= '/static'

    data = {'school': 'Hogwarts','student': student}  session['login'] = {'school': 'Hogwarts'}
    return render_template('register_student.html', **data)   # , student=student
    # this will file the html {{ school }} {{ session.login.school }} {{ student.name }} with content

    Filter process in html
    Welcome to {{school|reverse|upper|capitalize}}</h2> <!--{{school|capitalize}} title, trim    -->
    # default('value')   set default value if not specified
    {{<p>"hello, %s"</p>|format('harry)|striptags}}  # striptags show tags as string
    # safe  render html tags, won't do escape conversion,(treat as html code)   same as r''
    # escape  (treat as string python code)

     # unique  remove duplicate in list ignore case    # join(',')  use character to link list items to string
     # first/last  return first / last item in list     # random  return random item of list
     # slice(3)|list  divide list into small lists (3 items each) inside list wrapper   # sort
    # min / max / sum     return min/max/sum item in list

    # int / float  convert value to int / float      # round(precision = 0.1, method="common')  ceil,floor  round number
    # filesizeformat   convert to mb instead of byte

    # customized filter:
    @app.template_filter(datfmt')
    def datefmt_filter(value, *args):  return value.strftime(value, args[0])

    block structure in template
    base.html  add  {% block name_a %} default value {% endblock %}  at place for customized codes
    index.html   {% extends "base.html" %}  # extend parent page structure
    {% block name_a %} new value {% endblock %}  to replace base default value
    return render_template('index.html')     use {% block name_a %} {{ super() }} {% endblock %} to keep parent value
    {% include "content.html" %}    # include will bring every thing from content.html into current page
    content.html
    {% macro input(macroId, placeholder) %}
    <input type=text id="input_{{macroId}}" placeholder="{{placeholder}}" class="input_field"> {% endmacro %}
    index.html   # every additional {% %} {{}} code have to be inside {block}
    {% from "content.html" import input with context %}
    {{ input('name', 'name') }}
    {% for item in house %}
         <li {% if loop.index % 2 != 0 %}class="odd" {% endif %}> {{ item }}</li>{% endfor %}


    app = Flask('appName')
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        return "<p>Hello World</p>" / Response / render_template('register_student.html', **data) ...
    app.register_blueprint(emp.bp, url_prefix='/emp')  # put separate blue print for different model in the views folder
        # url_prefix optional, add additional in the routing path. request path match the rounting path
    app.run(host="localhost", port=5000, debug=True, threaded=True)
            # debug mode, change in code will restart server, will show error

    set response header (content type, default text/html)
    from flask import make_response, jsonify, Response
    emp.py
    bp = Blueprint('emp', __name__)  # use blue print to split the work into different class (employer)
        # inside main app class:
    @bp.route('/find/<int:id>', methods=['GET', 'POST'])  # <int:id>
    def employee(id):
        data = '{"id": 101, "age":  20}'
        response = make_response(data, 200)   # data need to be json (careful byte and date need to convert)
            # here make_response is from flask package not from function parameter (native python)
        response.headers['Content-Type'] = 'application/json;charset=utf-8'
        return response
        # return jsonify(data ,code) # jsonify already set 'application/json;charset=utf-8'
        # return Response(data, 200, content-type='application/json')  #

        # redirect start a new response
        # return redirect("%s" % url_for('emp.display', data=res))  #  , code=200  don't need specific display logic
        # return redirect('display', data=res)   return redirect(url_for('emp.display'))

        # request exception, data is wrong, stop request
        if not data:
            #abort(403)  # status code. return error page with status code
            abort(Response("data not valid", 403))  # response page with error information

    # catch exception
    @app.errorhandler(404)  # only catch 404 status code   # @app.errorhandler(Exception)  catch exception
    def notfound(err):
        print(err)
        return '404 page'

    Cookie: store data at client side(browser), browser setup storage space for each ip (host). Cookie stored as
        key = value pair. and can set up time to live for key. A complete Cookie contains: name, content, host
        response = make_response(data, 200)
        response.set_cookie('username','harry', expires= datetime.strptime('2022-10-31 16:55:00','%Y-%m-%d %H:%M:%S'))
            # or use max_age=10 (in second) instead of expires    add cookie  name, value, expires -1 forever
        response.delete_cookie('username')  # delete cookie
        cookie = request.cookies.get('username')  # get cookie value
        # chrome check cookie under Application tab

    Session: is a client and server connection under HTTP. HTTP 1.0, session is only lifecycle of one request and
        response. HTTP 1.1, session become long connection (header connection: keep alive), multiple requests and
        responses share one session. Create a session for each client side(browser). Use session to store data can be
        access by all requests, data like (user login information, verification, geography location)
        client side send cookie (inside it, have a session_id), use session_id to distinguish which session is for which
        user. flask session is based on cookie to store session id
        flask default session data store at client browser, use flask-session plugins to store session in database or
        cache(redis)
        
        app.secret_key = "super secret key"  # need add secret key to use sessions
        # app.config.from_object(settings.Dev) # use configure file
            # settings.py class Dev(): ENV='development';  SECRET_KEY='123456'
        from flask import sessions
        pwd = hashlib.md5('123456'.encode('utf-8')).hexdigest()
        login = [username, pwd]
        sessions['login'] = login   # bind login info to key 'login' save in session
        session.get('login')
        del sessions['login']

        can save session_id on server redis /file /db. but still client use cookie to access sessions
        from flask_session import Session
        settings.py    Dev():  # add redis config
        SESSION_TYPE = 'redis'
        SESSION_REDIS = Redis(host='127.0.0.1', db=1)
        server_session = Session()  # flask_session.Session, used for save session in server db/cache.
        server_session.init_app(app)

    ORM (object relationship mapping)
    DAO structure eased the operation on database, but requires knowledge on SQL and python sql package to operate
        correctly and avoid SQL injection
    using ORM link models(entries) to database tables, instances to rows, attributes to columns. operations on models
        is equivalent to operations in database

"""
import sys
from wsgiref.simple_server import make_server
import os
from flask import Flask, render_template, jsonify, make_response, session
from flask import request
from flask.cli import FlaskGroup
from flask_session import Session
import settings
from views import emp
from datetime import datetime

class Student:
    def __init__(self, name, age):
        self.name= name
        self.age=age
def app(env, make_response):
    """
        PATH_INFO  (request path, start with /)
        REQUEST_METHOD (get, post, patch, delete, update)
        QUERY_STRING (after ?  country=united+states)
        REMOTE_ADDR (client ip)
        CONTENT_TYPE (request data type)
        HTTP_USER_AGENT  (client agent (browser))
        wsgi.input (request byte object)
        wsgi.multithread  (whether use multi-thread)
        wsgi.multiprocess  (whether use multi-process)
    """
    # for k, v in env.items():
    #    print(k, ':', v)

    # from flask_cors import CORS
    # CORS().init_app(app)   #for CORS error   Cross-origin resource sharing

    path = env.get('PATH_INFO')
    header = []  # response header  tuple pair
    body = []
    static_dir = os.path.join('..', 'resources')  # basic dir is current file's parent
    if any((path.endswith('.png'), path.endswith('.jpg'), path.endswith('.gif'))):
        if path.index('resources') != -1:  # path: '/resources/images/hp.jpg
            res_path = os.path.join('..', *path.split('/'))
        else:
            res_path = os.path.join(static_dir, 'images', path[1:])
        header.append(('content-type', 'images\*'))
    elif path == '/':
        res_path = 'html_css_js_notes.html'
        header.append(('content-type', 'text/html;charset=utf-8'))
    elif path.endswith('.js'):
        if path.index('resources') != -1:  # path: '/resources/images/hp.jpg
            res_path = os.path.join('..', *path.split('/'))
        else:
            res_path = os.path.join(static_dir, 'js', path[1:])
        header.append(('content-type', 'text/*;charset=utf-8'))
    elif path.endswith('.html'):
        res_path = os.path.join(path[1:])
        header.append(('content-type', 'text/html;charset=utf-8'))
    else:
        if path.index('resources') != -1:  # path: '/resources/css/style.css
            res_path = os.path.join('..', *path.split('/'))
        else:
            res_path = os.path.join(static_dir, path[1:])
        if path.endswith('.mp3'):
            header.append(('content-type', 'audio/mpeg'))
        if path.endswith('.mp4'):
            header.append(('content-type', 'video/mp4'))

    status_code = 200
    if not os.path.exists(res_path):
        status_code = 404
        body.append('<h4 style="color:red">resource requested does not exist:404</h4>'.encode('utf-8'))
    else:
        with open(res_path, 'rb') as f:
            body.append((f.read()))
    make_response('%s OK' % status_code, header)  # response header
    return body


def app2():



    # create Flask object (Httpd web service object)
    app = Flask('harrypotter')  # app = Flask(__name__)   name need to be lowercase
    # method can be 'GET'  'POST'  'PUT'  'DELETE' (RESTFUL service action)
    # request path can have parameter, use converter (string, path, int, float, uuid, any) to get parameter
    # default parameter is string if not specified   @app.route('/find/word',methods=['GET', 'POST'])
    # @app.route('/find/<int:id>', methods=['GET', 'POST'])
    # def find(word)
    # @app.route('/forward/<path:url>', methods=['GET', 'POST'])
    # def forward(url):   return redirect(url)     /forward/http://www/baidu.com  is legal for path, not string

    # url_for('blue_print_name.function_name', **kwargs)  or url_for('function_name', **kwargs)
    #  @bp.route('/show/<data>')
    #  def display(data):
    #      return redirect("%s" % url_for('emp.display', data=res))

    # no need write out the path (emp/show/<data>), dynamically generate url_prefix
    # data is the input parameter of display request path

    app.config.from_object(settings.Dev)   # read flask config from file


    @app.route('/login', methods=['GET', 'POST'])
    def login():
        # request (HttpRequest) include request path, method, header, form data, file
        # get parameter in url after ?
        magic = request.args.get('magic', 'no')  # default value is pc if not retrieved any data
        cookie = request.cookies.get('username')  # get cookie value
        print(cookie)
        if magic.lower() != 'yes':
            return '''<h2>Use path: login?magic=yes href="/login">Retry</a></h2>  '''
        if request.method == 'GET':
            return '''<h1> User Login </h1>
                            <form action="/login?magic=yes" method="post">
                                <p>username: <input name="name" type="text"></p>
                                <p>password: <input name="pwd" type="password"></p>
                                <button>Submit</button>
                            </form>  
                            '''
        else:
            name = request.form.get('name')  # get parameter in post method form
            pwd = request.form.get('pwd')
            if name == 'harry' and pwd == '123456':
                return '''<h2>Login successfully</h2>  '''
            else:
                return '''<h2>Login failed. <a href="/login?magic=yes">Retry</a></h2>  '''

    """
        bp = Blueprint('emp', __name__)
        @bp.route('/find', methods=['GET', 'POST'])  # config routing  
        def employee():
            return "Hi"
    """

    @app.route('/hi', methods=['GET', 'POST'])
    def hi():
        data = 'Hi'
        response = make_response(data, 200)  # data need to be json (careful byte and date need to convert)
            # default status code is 200, so same as make_response(data)
        response.headers['Content-Type'] = 'text/html' # default value
        response.set_cookie('username','harry', expires=datetime.strptime('2022-10-31 16:55:00','%Y-%m-%d %H:%M:%S'))
            # or use max_age=10 (in second) instead of expires    add cookie  name, value, expires -1 forever
        response.delete_cookie('username')  # delete cookie
        return response

    @app.route('/', methods=['GET', 'POST'])
    def index():
        student = Student('Harry Potter', 10)
        data = {'house': ["Gryffindor","Hufflepuff","Ravenclaw","Slytherin"],'app':'app','student':student}

        session['login'] = {'school': 'Hogwarts'}
        return render_template('index.html', **data)

    @app.errorhandler(404)
    def notfound(err):
        return '404 page'

    app.register_blueprint(emp.bp, url_prefix='/emp')  # put separate blue print for different model in the views folder
        # url_prefix optional, add additional in the routing path. request path match the rounting path
    return app


def app3():
    # app = Flask(__name__)  #  template_folder default is templates, same parent folder and directory name is templates

    template_folder = os.path.join('..', 'resources', 'templates')  # not recommended
    app = Flask(__name__, template_folder=template_folder)
    # app.secret_key = "super secret key"
    app.config.from_object(settings.Dev)  #  add secret_key, redis config
    server_session = Session()  # flask_session.Session, used for save session in server db/cache.
    server_session.init_app(app)

    @app.route('/stundent', methods=['GET', 'POST'])
    def regist_stundent():
        # load data from model

        data = {
            'school': 'hogwarts',
            'date': datetime.now(),
            'error_message': ''
        }
        #data = jsonify(data)  # jsonify({'school': 'Hogwarts','error_message': ''})

        # return html templates at template_folder
        if request.method == 'GET':
            return render_template('register_student.html', **data)  # this will file the html {{school}} with content
            #return render_template('register_student.html', **locals())
        else:
            school = request.form.get('school_name', None)
            student = request.form.get('student_name', None)

            if not student or not school:
                data['error_message'] = "school name and student name can't be empty"
                return render_template('register_student.html', **data)
            app.logger.info('student: %s -> school: %s' % (student, school))
            session['login'] = {'name': student,'school': school}
            return '''
                    <h2>student registered successful</h2>
                    <script>alert(" %s is registered")</script>
                    ''' % student

    @app.route('/hi', methods=['GET', 'POST'])
    def hi():

        return "Hi, %s. %s welcome you." % (session.get('login').get('name'),session.get('login').get('school'))


    return app


if __name__ == '__main__':
    opt = int(input('python server: enter option(1-3): 1. native python   2. flask   3. flask mtv'))
    if opt == 1:    # http://localhost:8000/
        # native python web server
        httpd = make_server('127.0.0.1', 8000, app)  # http daemon
        httpd.serve_forever(poll_interval=0.5)  # interval 0.5sec for listen response thread end

    elif opt == 2:    # http://localhost:5000/login?magic=yes
        app2 = app2()
        app2.run(host="localhost", port=5000)   # default single thread single process(extendable)
                                                # threaded=True  # processes=4   can't use both same time
    elif opt == 3:
        app3 = app3()  # http://localhost:5000/stundent
        app3.run(host="localhost", port=5000, debug=True, threaded=True)
            # debug mode, change in code will restart server, will show error
