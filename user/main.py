import random
import sys

from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget, QMainWindow)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import time
import pickle
import socket

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
import tensorflow as tf

import numpy as np


class Model:
    def __init__(self, graph_path, label_path):
        self.graph_path = graph_path
        self.label_path = label_path
        self.output_dict = {}
        self.tensor_dict = {}
        self.category_index = {}
        self.detection_thresh = 0.6
        self.image_np = np.ndarray((1, 1, 1), dtype=np.uint8)

        detection_graph = tf.Graph()
        detection_graph.as_default()
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        self.category_index = label_map_util.create_category_index_from_labelmap(
            label_path, use_display_name=True)

        self.session = tf.Session()

        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)

    def detect(self, image_np):
        self.image_np = image_np
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        self.output_dict = self.session.run(self.tensor_dict,
                                            feed_dict={image_tensor: image_np_expanded})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        self.output_dict['num_detections'] = int(
            self.output_dict['num_detections'][0])
        self.output_dict['detection_classes'] = self.output_dict[
            'detection_classes'][0].astype(np.uint8)
        self.output_dict['detection_boxes'] = self.output_dict['detection_boxes'][0]
        self.output_dict['detection_scores'] = self.output_dict['detection_scores'][0]

        self.num_hands = sum(
            self.output_dict['detection_scores'] >= self.detection_thresh)
        return self.num_hands

    def display_results(self):
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            self.image_np,
            self.output_dict['detection_boxes'],
            self.output_dict['detection_classes'],
            self.output_dict['detection_scores'],
            self.category_index,
            instance_masks=self.output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            min_score_thresh=self.detection_thresh,
            line_thickness=6)
        # Size, in inches, of the output image.
        disp_size = (24, 16)
        plt.figure(figsize=disp_size)
        plt.imshow(self.image_np)
        plt.show()


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

    def send_cap_trigger(self):
        # Sending socket setup
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_sock.connect((self.send_host, self.send_port))
        self.send_sock.sendall(b'cap')
        self.send_sock.close()

    def wait_image_data(self):
        data = b''
        self.conn, self.addr = self.recv_sock.accept()
        print('Incoming connection from:', self.addr)
        tic = time.time()
        while True:
            inc_data = self.conn.recv(8192)
            if inc_data == b'':
                print('Received {} bytes.'.format(len(data)))
                break
            else:
                data += inc_data
        toc = time.time()
        print('RECEIVE TIME: {:.3f}'.format(toc-tic))
        image_np = pickle.loads(data)
        return image_np


class Results:
    def __init__(self):
        self.results_dict = {'A': 0,
                             'B': 0,
                             'C': 0,
                             'D': 0,
                             'E': 0}

    def add_result(self, option, num_hands):
        self.results_dict[option] = num_hands

    def reset_results(self):
        self.results_dict = {'A': 0,
                             'B': 0,
                             'C': 0,
                             'D': 0,
                             'E': 0}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Axes
        self.MainWidget = QWidget(self)
        self.setCentralWidget(self.MainWidget)

        self.MainWidget.figure = plt.figure()

        self.MainWidget.canvas = FigureCanvas(self.MainWidget.figure)

        self.MainWidget.plot_button = QPushButton('Test Plot')
        self.MainWidget.plot_button.clicked.connect(self.test_plot)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.MainWidget.canvas, 2)
        main_layout.addWidget(self.MainWidget.plot_button)
        self.MainWidget.setLayout(main_layout)

        # Buttons
        self.ButtonWidget = QWidget(self)

        self.ButtonWidget.buttonA = QPushButton('A')
        self.ButtonWidget.buttonA.clicked.connect(self.pollA)

        self.ButtonWidget.buttonB = QPushButton('B')
        self.ButtonWidget.buttonB.clicked.connect(self.pollB)

        self.ButtonWidget.buttonC = QPushButton('C')
        self.ButtonWidget.buttonC.clicked.connect(self.pollC)

        self.ButtonWidget.buttonD = QPushButton('D')
        self.ButtonWidget.buttonD.clicked.connect(self.pollD)

        self.ButtonWidget.buttonE = QPushButton('E')
        self.ButtonWidget.buttonE.clicked.connect(self.pollE)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ButtonWidget.buttonA)
        button_layout.addWidget(self.ButtonWidget.buttonB)
        button_layout.addWidget(self.ButtonWidget.buttonC)
        button_layout.addWidget(self.ButtonWidget.buttonD)
        button_layout.addWidget(self.ButtonWidget.buttonE)
        self.ButtonWidget.setLayout(button_layout)

        main_layout.addWidget(self.ButtonWidget)

        self.MainWidget.reset_button = QPushButton('Reset')
        self.MainWidget.reset_button.clicked.connect(self.reset_poll)
        main_layout.addWidget(self.MainWidget.reset_button)

        graph_path = 'frozen_inference_graph.pb'
        label_path = 'label_map.pbtxt'

        self.model = Model(graph_path, label_path)

        self.options_list = ['A', 'B', 'C', 'D', 'E']

        self.connection = Connection(
            '10.42.0.1', 5001, '10.42.0.171', 5001)

        self.results = Results()

        # create an axis
        self.ax = self.MainWidget.figure.add_subplot(111)
        self.MainWidget.figure.clear()

    def pollA(self):
        option = 'A'
        self.connection.send_cap_trigger()
        inc_data = self.connection.wait_image_data()
        image_np = inc_data
        tic = time.time()
        num_hands = self.model.detect(image_np)
        toc = time.time()
        print('INFERENCE TIME: {:.3f}'.format(toc-tic))

        self.results.add_result(option, num_hands)
        self.plot()

    def pollB(self):
        option = 'B'
        return option

    def pollC(self):
        option = 'C'
        return option

    def pollD(self):
        option = 'D'
        return option

    def pollE(self):
        option = 'E'
        return option

    def reset_poll(self):
        self.results.reset_results()
        self.ax.clear()
        self.MainWidget.figure.clear()
        self.MainWidget.canvas.draw()

    def plot(self):
        self.MainWidget.figure.clear()

        # create an axis
        self.ax = self.MainWidget.figure.add_subplot(111)

        results_dict = self.results.results_dict
        results = [results_dict[option] for option in results_dict]
        self.ax.bar(self.options_list, results, color='deepskyblue')

        rects = self.ax.patches
        labels = results
        for rect, label in zip(rects, labels):
            if label == 0:
                pass
            else:
                height = rect.get_height()
                self.ax.text(rect.get_x() + rect.get_width() / 2,
                             height / 2, label, ha='center', va='top')

        # refresh canvas
        self.MainWidget.canvas.draw()

    def test_plot(self):

        self.MainWidget.figure.clear()

        # create an axis
        self.ax = self.MainWidget.figure.add_subplot(111)

        results = [random.randint(0, 10) for _ in range(5)]
        self.ax.bar(self.options_list, results, color='deepskyblue')

        rects = self.ax.patches
        labels = results
        for rect, label in zip(rects, labels):
            if label == 0:
                pass
            else:
                height = rect.get_height()
                self.ax.text(rect.get_x() + rect.get_width() / 2,
                             height / 2, label, ha='center', va='top')

        # refresh canvas
        self.MainWidget.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = MainWindow()
    main.show()

    sys.exit(app.exec_())
