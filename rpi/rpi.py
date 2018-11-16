import socket
import pickle
import time
import picamera
import io
from PIL import Image
import numpy as np

class Connection:
    def __init__(self, recv_host, recv_port, send_host, send_port):
        self.recv_host = recv_host
        self.recv_port = recv_port

        self.send_host = send_host
        self.send_port = send_port

        # Receiving socket setup
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_sock.bind((self.recv_host, self.recv_port))
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


stream = io.BytesIO()
cam = picamera.PiCamera()
cam.resolution = (1920, 1080)

connection = Connection('10.42.0.171', 5001, '10.42.0.1', 5001)

while True:
    data = connection.wait_data()
    if data == b'cap':
        # for _ in range(50):
        cam.capture(stream, format='jpeg')
        stream.seek(0)
        image_pil = Image.open(stream)
        image_np = np.array(image_pil)
        image_data = pickle.dumps(image_np)
        connection.send_image(image_data)
    else:
        print('Bad communication.')
        break


# tic = time.time()


# data_dict = {'type': 'image',
#              'image': image_np,
#              'option': 'A'}

# data = pickle.dumps(data_dict)

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((host, port))

# s.sendall(data)
# s.close()

# toc = time.time()
# print('ELAPSED TIME:', toc-tic)
