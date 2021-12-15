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


    # native wsgi implementation
    def app1(env, make_response):
        header, body=[],[]
        request_path = env.get('PATH_INFO')
        header.append(('content-type', 'images\*'))'text/html;charset=utf-8'  'audio/mpeg'   'video/mp4'
        make_response('%s OK' % status_code, header)  # native python implementing wsgi,
        return body  # content inside body (list, string)
    httpd = make_server('127.0.0.1', 8000, app1)  # http daemon
    httpd.serve_forever(poll_interval=0.5)


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
      request.get_json()
    both request and response have headers and bodies
    response = requests.request(method, url)   # requests are used for testing, simulate browser send request
    response is the return object in the views functions

    from flask import Flask, render_template, make_response, session, jsonify   (based on Jinja2 package)
    template_folder = os.path.join('..', 'resources', 'templates')
    app = Flask(__name__, template_folder=template_folder, static_folder='resource')  # set templates folder location,
        if not under same parent and folder name is templates. static default folder is same parent, and named static
        static_url_path='static'  default is /    # add request prefix for static folder item
        <link rel="stylesheet" href="/static/css/my.css">  to access the css file in resource/css folder
        same as set in config file: STATIC_FOLDER = '/resource'   STATIC_URL_PATH = '/static'
        or app.static_url_path= '/static'


    data = {'school': 'Hogwarts','student': student}  session['login'] = {'school': 'Hogwarts'}
    return render_template('register_student.html', **data)   # , student=student
    # this will fill the register_student.html {{ school }} {{ session.login.school }} {{ student.name }} with content

    tests need add app.app_context().push()

    Filter process in html
    Welcome to {{school|reverse|upper|capitalize}}</h2> <!--{{school|capitalize}} title, trim    -->
    # {{school|default('value')}}   set default value if not specified
    {{<p>"hello, %s"</p>|format('harry')|striptags}}  # striptags show tags as string
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
    {% macro input(macroId, placeholder) %}   # macro template function
    <input type=text id="input_{{macroId}}" placeholder="{{placeholder}}" class="input_field"> {% endmacro %}
    {{ input('user_name','Harry Potter') }}   # use macro

    index.html   # every additional {% %} {{}} code have to be inside {block}
    {% from "content.html" import input with context %}
    {{ input('name', 'name') }} # call this macro inside other html template
    {% for item in house %}
         <li {% if loop.index % 2 != 0 %}class="odd" {% endif %}> {{ item }}</li>{% endfor %}


    app = Flask('appName')
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        return "<p>Hello World</p>" / Response / render_template('register_student.html', **data) ...
    app.register_blueprint(emp.bp, url_prefix='/emp')  # put separate blue print for different model in the views folder
        # url_prefix optional, add additional in the routing path. request path match the routing path
        # able to visit additional blueprint
    app.run(host="localhost", port=5000, debug=True, threaded=True)
            # debug mode, change in code will restart server, will show error

    set response header (content type, default text/html)
    from flask import make_response, jsonify, Response, Blueprint
    emp.py
    bp = Blueprint('emp', __name__)  # use blue print to split the work into different class (employer)
        # inside main app class:
    @bp.route('/find/<int:id>', methods=['GET', 'POST'])  # <int:id>
    def employee(id):
        data = '{"id": 101, "age":  20}'
        response = make_response(data, 200)   # data need to be json (careful byte and date need to convert)
            # make_response is from flask package not from function parameter (native python), return Response object
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

    Reversing namespaced URLs: get url from path logic
    # url_for('blue_print_name.function_name', **kwargs)  or url_for('function_name', **kwargs)
    #  @bp.route('/show/<data>')
    #  def display(data):
    #      return redirect("%s" % url_for('emp.display', data=res))

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
        # app.config.from_object(settings.Dev) # or use configure file
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
        inside app()
        server_session = Session()  # flask_session.Session, used for save session in server db/cache.
        server_session.init_app(app)

    ORM (object relationship mapping)
    DAO structure eased the operation on database, but requires knowledge on SQL and python sql package to operate
        correctly and avoid SQL injection
    using ORM link models(entries) to database tables, instances to rows, attributes to columns. operations on models
        is equivalent to operations in database
    DAO
    def save(entity):
        print(dir(entity),[a for a in dir(entity) if not a.startswith('__')])  # print object normal attribute
        print(entity.__dict__, {a:getattr(entity,a) for a indir(entity) if not a.startswith('__')}}
        sql = "insert into %s(%s) values (%s)"
        table = entity.__class__.__name__.lower()
        columns = ','.join([c for c in user.__dict__])
        placeholder = ','.join(["%%(%s)s" % c for c in user.__dict__])
        sql = sql % (table, columns, placeholder)

    check package __init__ for configuration
    link to database path: dialect+driver://user:password@ip:port/db?charset=utf-8
    inside settings.DEV  add
        SQLALCHEMY_DATABASE_URI = 'mysql+pymsql://cai:123456@127.0.0.1:3306/company?charset=utf-8'
        SQLALCHEMY_TRACK_MODIFICATIONS = True      # coexist with previous version
        SQLALCHEMY_COMMIT_ON_TEARDOWN = True  # commit transaction when destroy resources
        SQLALCHEMY_ECHO = True       # display debug sql message

    same as app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://cai:123456@127.0.0.1:3306/company'

    inside  __init__.py for models directory
    from flask-sqlalchemy import SQLAlchemy
    db = SQLAlchemy()

    entity class
    class Emp(db.Model): #default class name is table name
        d_id = db.column(db.Integer, primary_key=True)  # default column name is attribute name  #, autoincrement=True
        name = db.column('d_name',db.String(50))  # default column name is attribute name, add 'd_name' to change name
        d_address = db.column(db.String(100))     #       unique=True, nullable=False
        e_dept = db.Column(db.Integer, db.ForeignKey('Dept.id'))
            # server_default=text('NOW()') default value for table      default=0  default for model attribute
        # db.Text  db.Unicode  db.Date  db.DateTime   db.Boolean  db.Float

    class BaseModel(db.Model):  parent class used for extend
        __abstract__ = True  # won't create table
        id = Column(Integer, primary_key=True,autoincrement=True)
    class Child(BaseModel):
        __tablename__='role'  # change table name

    main script
    from models.dept import Dept
    from models import db
    app.config.from_object(settings.Dev)   # read flask config from file
    db.init_app(app)
    db.create_all()   # create table for models automatically  need import model before calling method
    db.drop_all()   # delete table for models

    d = Dept(10,'a','Hogwarts')
    db.session.add(d)  # insert
    d.name = 'Gryffindor'  # update value, no need to commit
    db.session.delete(Dept.query.get(10))  #delete
    db.session.commit()
    Dept.query.get(10)  # query by id
    Dept.query.all()   # find all       Dept.query.count()   # total counts
    print(Dept.query.filter_by(d_name='Information Technology').one())  # .one cause exception if no result found
    print(Dept.query.filter(db.or_(Dept.d_name.startswith('Inf'), Dept.d_name == 'magic')).all())  # db.not_
    for d in Dept.query.filter(Dept.d_name.contains('ryf')):  # startswith  endswith
    for d in Dept.query.filter(Dept.id.__ge__(5)):  # startswith  endswith

    for d in db.session.query(Dept).all()  # return list of object
    db.session.query(Dept.id,Dept.d_name)  # return list of namedtuple
    for d in db.session.query(Dept.id,Dept.d_name).filter(Dept.d_name.like('%ryf%')).all():  # one()
        # session search, only for specific columns    filter use all(),one()     query use first(), all(), get(id)
    order_by(Dept.d_name, Dept.d_address.desc())  # order by name if same then order by address, default asc()
        can be used  after session.query and query, before or after filter
    for d in Dept.query.order_by(Dept.id).offset(3).limit(3).all():  # paging   limit offset
        can be used  after session.query and query, before or after filter, must after order_by
    # from sqlalchemy import func
    count = db.session.query(db.func.count(Dept.d_id)).first()
    count = db.session.query(Dept.d_name, db.func.count(Dept.d_id).label('cnt')).group_by(Dept.d_name)
        .having(db.func.sum(db.Column('cnt').__ge__(2)).order_by(db.Column('cnt').desc()).all()
        # label: add alias  group_by return tuple

    join search
    db.session.query(t_emp.e_name, Dept.d_name).filter(t_emp.e_dept == Dept.d_id)
    join()  outerjoin()  need foreign_keys

    db.Column() methods:  label()  desc()  asc()  startswith()  endswith()  like()  contains()  le()  lt()
        ge()  gt()  eq()  in_()  notin_()  isnot()

    relationship
    setup foreign keys/ relation at many side table/entity
    emp:
        e_dept = db.Column(db.Integer, db.ForeignKey('Dept.id'))
    Dept:
        dept = db.relationship(Dept, backref='emps')  # backref inverse search
            #  db.relationship(Dept, backref=db.backref('emps', lazy=True))  # default is lazy load
                # lazy=False will use outer join fetch both table
    dept = Dept.query.get(1)
    print(dept.emps)  #  inverse search, specified in relationship parameter
    for emp in dept.emps:
        print(emp.dept.d_name)   # dept specified in relationship output

    for many to many relationship, create table for relationship
    emp_dept = db.Table('emp_dept', Column('e_id',db.Integer, ForeignKey('t_emp.e_id')), Column('d_id',db.Integer,
        ForeignKey('Dept.id')) )   # 'emp_dept' table name
    emp:
        depts = db.relationship(Dept, secondary=emp_dept)  # secondary for many to many relationship table


    app.logger.info('student: %s -> school: %s' % (student, school))
    noset, debug, info, warning, error, critical
    can delete flask logger and use customized handler:
        logging.StreamHandler/FileHandler       logging.handlers.HttpHandler/SMTPHandler

        logger = logging.getLogger('learn_api')
        def config_log():
            fmt = Formatter(fmt='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
            io_handler = StreamHandler()
            file_handler = FileHandler('app.log')  # file name for logging
            file_handler.setLevel(logging.WARN)
            file_handler.setFormatter(fmt)
            http_handler = HTTPHandler(host='localhost:5000', url='/log',method='POST')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(http_handler)
        config_log()
        logger.info('test')  # warning  error  critical

        app.logger.removeHandler(default_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)


    cache use redis or flask Cache decorators

    from flask_caching import Cache
    cache = Cache(app,config={'CACHE_TYPE': 'simple'})  # 'redis'
        or in settings.py add #CACHE_TYPE = 'redis'   #CACHE_REDIS_HOST = '127.0.0.1'
            #CACHE_REDIS_PORT = 6379   #CACHE_REDIS_DB = 5
            app.config.from_object(settings.Dev)
    @cache.cached(timeout=xxx)   @app.route('/', methods=['GET', 'POST'])  def index():
    @cache.cached(timeout=None, key_prefix='all_comments')  # for non view functions

    cache functions:  init_app(app)   get()  set()  add()  get_many()  set_many()  delete_many()  clear()
    cache.set('user','Harry')


    @app.before_first_request
    @app.before_request
    @app.after_request
    @app.teardown_request

    @app.after_request     # specify common logic after receiving a request
    def foot_log(environ):
        if request.path != "/login":
            print("some one visited",request.path)
        return environ


    # from flask_cors import CORS
    # CORS().init_app(app)   #for CORS error   Cross-origin resource sharing

    flask-RESTful
    from flask_restful import Api,Resource
    app = Flask(__name__)
    api = Api(app)

    class IndexView(Resource):
        def get(self):     # post patch put delete
            return {"username":"harry"}
    api.add_resource(IndexView,'/',endpoint='index')

    flask-sqlacodegen

    add \ to switch line


    environment
    virualenv (small)   conda   docker


    gunicorn server for linux system, windows not support
    gunicorn --config gunicorn.conf main:app

"""
import uuid
from wsgiref.simple_server import make_server
import os
from flask import Flask, render_template, make_response, session, jsonify
from flask import request
from flask_session import Session
import settings
from views import emp
from datetime import datetime


class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age


def app1(env, make_response):
    """
        env dict contains:
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
    from models.dept import Dept

    app.config.from_object(settings.Dev)  # read flask config from file
    from models import db
    # app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://cai:123456@127.0.0.1:3306/company'
    db.init_app(app)

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
        response.headers['Content-Type'] = 'text/html'  # default value
        response.set_cookie('username', 'harry', expires=datetime.strptime('2022-10-31 16:55:00', '%Y-%m-%d %H:%M:%S'))
        # or use max_age=10 (in second) instead of expires    add cookie  name, value, expires -1 forever
        response.delete_cookie('username')  # delete cookie
        return response

    @app.route('/', methods=['GET', 'POST'])
    def index():
        student = Student('Harry Potter', 10)
        data = {'house': ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"], 'student': student,
                'dept': Dept.query.all()}
        session['login'] = {'school': 'Hogwarts'}

        dept = Dept(10, 'a', 'Hogwarts')
        db.session.add(dept)
        db.session.commit()
        dept.d_name = 'Gryffindor'  # update commit automatically
        db.session.delete(Dept.query.get(10))
        db.session.commit()
        print(Dept.query.filter_by(d_name='Information Technology').one())  # .one cause exception if no result found
        print(Dept.query.filter(db.or_(Dept.d_name.startswith('Inf'), Dept.d_name == 'magic')).all())
        # db.not_   db.and_
        for d in Dept.query.filter(Dept.d_name.contains('ryf')):  # startswith  endswith
            print(d)
        for d in Dept.query.filter(Dept.id.__ge__(5)):  # startswith  endswith
            print(d)
        for d in db.session.query(Dept.id, Dept.d_name).filter(Dept.d_name.like('%ryf%')).all():  # session search
            print(d)
        for d in Dept.query.order_by(Dept.id).offset(3).limit(3).all():  # paging
            print(d)
        print(db.session.query(db.func.count(Dept.id)).first())
        print(db.session.query(Dept.d_name, db.func.count(Dept.id).label('cnt')).group_by(Dept.d_name) \
              .having(db.func.sum(Dept.id).__ge__(2)).order_by(db.Column('cnt').desc()).all())
        sql = "select d_id,d_name,d_address from t_dept where d_name = '%s' limit 3"  # native sql for complicate query
        for d_id, name, addr in db.session.execute(sql % 'Gryffindor', ).cursor:
            # pymysql.cursors.Cursor can be iterated, optional add fetchall()  fetchone()
            print(d_id, name, addr)

        token = uuid.uuid4().hex  # Generate a random UUID. hexadecimal uuid
        print(token)
        return render_template('index.html', **data)

    @app.errorhandler(Exception)
    def exception_found(exception):
        print(exception)
        return jsonify({'msg': str(exception)})


    @app.errorhandler(404)
    def not_found(err):
        print(err)

        return '404 page'

    app.register_blueprint(emp.bp, url_prefix='/emp')  # put separate blue print for different model in the views folder
    # url_prefix optional, add additional in the routing path. request path match the rounting path
    return app


def app3():
    # app = Flask(__name__)  #  template_folder default is templates, same parent folder and directory name is templates

    template_folder = os.path.join('..', 'resources', 'templates')  # not recommended
    app = Flask(__name__, template_folder=template_folder)
    # app.secret_key = "super secret key"
    app.config.from_object(settings.Dev)  # add secret_key, redis config
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
        # data = jsonify(data)  # jsonify({'school': 'Hogwarts','error_message': ''})

        # return html templates at template_folder
        if request.method == 'GET':
            return render_template('register_student.html', **data)  # this will file the html {{school}} with content
            # return render_template('register_student.html', **locals())
        else:
            school = request.form.get('school_name', None)
            student = request.form.get('student_name', None)

            if not student or not school:
                data['error_message'] = "school name and student name can't be empty"
                return render_template('register_student.html', **data)
            app.logger.info('student: %s -> school: %s' % (student, school))
            session['login'] = {'name': student, 'school': school}
            return '''
                    <h2>student registered successful</h2>
                    <script>alert(" %s is registered")</script>
                    ''' % student

    @app.route('/hi', methods=['GET', 'POST'])
    def hi():

        return "Hi, %s. %s welcome you." % (session.get('login').get('name'), session.get('login').get('school'))

    return app


if __name__ == '__main__':
    opt = int(input('python server: enter option(1-3): 1. native python   2. flask   3. flask mtv'))
    if opt == 1:  # http://localhost:8000/
        # native python web server
        httpd = make_server('127.0.0.1', 8000, app1)  # http daemon
        httpd.serve_forever(poll_interval=0.5)  # interval 0.5sec for listen response thread end

    elif opt == 2:  # http://localhost:5000/login?magic=yes
        app = app2()
        app.run(host="localhost", port=5000)  # default single thread single process(extendable)
        # threaded=True  # processes=4   can't use both same time
    elif opt == 3:
        app = app3()  # http://localhost:5000/stundent
        app.run(host="localhost", port=5000, debug=True, threaded=True)
        # debug mode, change in code will restart server, will show error
