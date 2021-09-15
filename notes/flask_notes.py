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

    Flask use M (model) T (template, html) V (views, controller)  which is based on MVC

"""
import sys
from wsgiref.simple_server import make_server
import os
from flask import Flask, render_template, jsonify
from flask import request
from flask.cli import FlaskGroup
from views import emp

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
    # @app.route('/find/<string:word>', methods=['GET', 'POST'])
    # def find(word)
    # @app.route('/forward/<path:url>', methods=['GET', 'POST'])
    # def forward(url):   return redirect(url)     /forward/http://www/baidu.com  is legal for path, not string

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        # request (HttpRequest) include request path, method, header, form data, file
        # get parameter in url after ?
        magic = request.args.get('magic', 'no')  # default value is pc if not retrieved any data
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

    app.register_blueprint(emp.bp, url_prefix='/emp')  # put separate blue print for different model in the views folder
        # url_prefix optional, add additional in the routing path. request path match the rounting path
    return app


def app3():
    # app = Flask(__name__)  #  template_folder default is templates, same parent folder and directory name is templates

    template_folder = os.path.join('..', 'resources', 'templates')  # not recommended
    app = Flask(__name__, template_folder=template_folder)

    @app.route('/stundent', methods=['GET', 'POST'])
    def regist_stundent():
        # load data from model

        data = {
            'school': 'Hogwarts',
            'error_message': ''
        }
        data = jsonify(data)  # jsonify({'school': 'Hogwarts','error_message': ''})

        # return html template at template_folder
        if request.method == 'GET':
            return render_template('register_student.html', **data)  # this will file the html {{school}} with content
        else:
            school = request.form.get('school_name', None)
            student = request.form.get('student_name', None)
            if not student or not school:
                data['error_message'] = "school name and student name can't be empty"
                return render_template('register_student.html', **data)
            app.logger.info('student: %s -> school: %s' % (student, school))
            return '''
                    <h2>student registered successful</h2>
                    <script>alert(" %s is registered")</script>
                    ''' % student




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
