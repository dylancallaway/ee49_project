import socket
import pickle
import time
import pygame
import pygame.camera


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


pygame.camera.init()
cam = pygame.camera.Camera('/dev/video0', (640, 480))
cam.start()

connection = Connection('10.142.184.27', 5002, '10.142.184.27', 5001)

while True:
    data = connection.wait_data()
    if data == b'cap':
        # for _ in range(50):
        surf_img = cam.get_image()
        image_np = pygame.surfarray.array3d(surf_img)
        image_np = image_np.swapaxes(0, 1)
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
