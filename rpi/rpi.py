#!/home/pi/ee49_project/rpi/venv/bin/python

import socket
import pickle
import time
import picamera
import io

class Connection:
    def __init__(self, recv_host, recv_port, send_host, send_port):
        self.recv_host = recv_host
        self.recv_port = recv_port

        self.send_host = send_host
        self.send_port = send_port

        # Receiving socket setup
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        for _ in range(20):
            try:
                self.recv_sock.bind((self.recv_host, self.recv_port))
                break
            except OSError:
                time.sleep(5)
            
        print('Listening at: {}:{}'.format(
            self.recv_host, str(self.recv_port)))
        self.recv_sock.listen(1)

    def send_image(self, data):
        # Sending socket setup
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_sock.connect((self.send_host, self.send_port))
        self.send_sock.sendall(data)
        self.send_sock.close()

    def wait_data(self):
        data = b''
        self.conn, self.addr = self.recv_sock.accept()
        print('Incoming connection from:', self.addr)
        data = self.conn.recv(3)
        print('Received {} bytes.'.format(len(data)))
        return data

    def end_connection(self):
        self.recv_sock.close()
        self.send_sock.close()


cam = picamera.PiCamera()
cam.resolution = (2592, 1944)

local_recv_host = '10.42.0.171'
local_send_host = '10.42.0.1'


connection = Connection(local_recv_host, 5001, local_send_host, 5001)

stream = io.BytesIO()

while True:
    data = connection.wait_data()
    if data == b'cap':
        # tic = time.time()
        cam.capture(stream, format='jpeg')
        stream.seek(0)
        image_data = pickle.dumps(stream)
        connection.send_image(image_data)
        # toc = time.time()
        # print('SEND TIME:', toc-tic)
    else:
        print('Bad communication.')
        break

connection.end_connection()



