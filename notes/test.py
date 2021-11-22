import time

import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('localhost', 8001))
sock.listen(5)
connection,address = sock.accept()
while True:
    try:
        connection.send(bytes('welcome to server!','utf-8'))
        time.sleep(1)
    except socket.timeout:
        connection.send(bytes('time out!', 'utf-8'))
        print('time out')

