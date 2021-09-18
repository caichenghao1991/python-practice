from redis import Redis
class Dev():
    ENV = 'development'
    SECRET_KEY='123456'

    SESSION_TYPE = 'redis'
    SESSION_REDIS = Redis(host='127.0.0.1', db=1)


    # SESSION_FILE_DIR = 'path'

    # STATIC_URL_PARH = '/static'
    # STATIC_FOLDER = '/resource'