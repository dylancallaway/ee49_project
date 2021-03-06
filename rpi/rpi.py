#!/home/pi/ee49_project/rpi/venv/bin/python

import socket
import pickle
import time
import picamera
import io
from gpiozero import LED


class Connection:
    def __init__(self, recv_host, recv_port, send_host, send_port):
        self.recv_host = recv_host
        self.recv_port = recv_port

        self.send_host = send_host
        self.send_port = send_port

        self.status = 0

    def wait_conn(self, num_mins):
        for _ in range(12*num_mins):
            try:
                connection.init_recv()
                self.set_status(1)
                break
            except OSError:
                print('Waiting for connection...')
                time.sleep(5)

    def send_image(self, trigger):
        # Sending socket setup
        try:
            self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.send_sock.connect((self.send_host, self.send_port))
            self.send_sock.sendall(trigger)
            self.send_sock.close()
        except OSError:
            self.set_status(0)
            self.send_sock.close()

    def init_recv(self):
        # Receiving socket setup
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_sock.bind((self.recv_host, self.recv_port))
        print('Connection successful.')
        print('Listening at: {}:{}'.format(
            self.recv_host, str(self.recv_port)))
        self.recv_sock.listen(1)

    def wait_trigger(self):
        trigger = b''
        self.conn, self.addr = self.recv_sock.accept()
        print('Incoming connection from:', self.addr)
        trigger = self.conn.recv(3)
        print('Received {} bytes.'.format(len(trigger)))
        return trigger

    def get_status(self):
        # 1 = OK, 0 = NOK
        return self.status

    def set_status(self, status):
        # 1 = OK, 0 = NOK
        self.status = status

    def end_connection(self):
        self.recv_sock.close()
        self.send_sock.close()
        connection.set_status(0)


led = LED(17)

local_recv_host = '10.42.0.171'
local_send_host = '10.42.0.1'

connection = Connection(local_recv_host, 5001, local_send_host, 5001)

stream = io.BytesIO()

cam = picamera.PiCamera(resolution=(2592, 1944))

while True:
    if connection.get_status() == 0:
        led.blink()
        connection.wait_conn(5)

    led.off()

    trigger = connection.wait_trigger()
    if trigger == b'cap':
        led.on()
        # tic = time.time()
        cam.capture(stream, format='jpeg')
        image_data = pickle.dumps(stream)
        connection.send_image(image_data)
        # toc = time.time()
        # print('SEND TIME:', toc-tic)
        stream.seek(0)
    else:
        connection.end_connection()
        cam.close()
        print('Bad communication. Connection closed.')
        break
