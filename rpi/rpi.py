#!/home/pi/ee49_project/rpi/venv/bin/python

import socket
import pickle
import time
import picamera
import io

HOST = '10.42.0.1'    # The remote host
PORT = 50007              # The same port as used by the server

stream = io.BytesIO()

cam = picamera.PiCamera(resolution=(2592, 1944))

cam.capture(stream, format='jpeg')


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        data = s.recv(1024)
        if len(data) > 1:
            cam.capture(stream, format='jpeg')
            data = pickle.dumps(stream.read(), 0)
            s.sendall(data)

stream.seek(0)
