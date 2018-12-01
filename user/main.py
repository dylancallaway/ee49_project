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
from PIL import Image


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
        image_data = pickle.loads(data)
        image_pil = Image.open(image_data)
        image_np = np.array(image_pil)
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
        self.title = 'AutoPoll'

        self.initUI()
        self.initModel()
        self.initConn()
        self.initResults()

    def initUI(self):
        self.setWindowTitle(self.title)

        # Results axes widget
        self.ResultsAxesWidget = QWidget(self)
        self.ResultsAxesWidget.figure = plt.figure(tight_layout=True)
        self.ResultsAxesWidget.canvas = FigureCanvas(
            self.ResultsAxesWidget.figure)
        results_axes_layout = QVBoxLayout()
        results_axes_layout.addWidget(self.ResultsAxesWidget.canvas)
        self.ResultsAxesWidget.setLayout(results_axes_layout)

        # Image axes widget in separate window (QDialog)
        self.ViewImageWindow = QMainWindow()
        self.ViewImageWindow.setWindowTitle('Results Image Display')
        self.ViewImageWindow.ImageAxesWidget = QWidget(self.ViewImageWindow)
        self.ViewImageWindow.ImageAxesWidget.figure = plt.figure(
            tight_layout=True)
        self.ViewImageWindow.ImageAxesWidget.canvas = FigureCanvas(
            self.ViewImageWindow.ImageAxesWidget.figure)
        self.ViewImageWindow.ImageAxesWidget.toolbar = NavigationToolbar(
            self.ViewImageWindow.ImageAxesWidget.canvas, self.ViewImageWindow.ImageAxesWidget)
        image_axes_layout = QVBoxLayout()
        image_axes_layout.addWidget(
            self.ViewImageWindow.ImageAxesWidget.canvas)
        image_axes_layout.addWidget(
            self.ViewImageWindow.ImageAxesWidget.toolbar)
        self.ViewImageWindow.ImageAxesWidget.setLayout(image_axes_layout)
        self.ViewImageWindow.setCentralWidget(
            self.ViewImageWindow.ImageAxesWidget)
        # self.ViewImageWindow.ImageAxesWidget.resize(
        #     self.ViewImageWindow.ImageAxesWidget.sizeHint())

        # Option buttons VBox
        self.OptionsWidget = QWidget(self)
        self.OptionsWidget.buttonA = QPushButton('A')
        self.OptionsWidget.buttonA.clicked.connect(self.pollA)
        self.OptionsWidget.buttonB = QPushButton('B')
        self.OptionsWidget.buttonB.clicked.connect(self.pollB)
        self.OptionsWidget.buttonC = QPushButton('C')
        self.OptionsWidget.buttonC.clicked.connect(self.pollC)
        self.OptionsWidget.buttonD = QPushButton('D')
        self.OptionsWidget.buttonD.clicked.connect(self.pollD)
        self.OptionsWidget.buttonE = QPushButton('E')
        self.OptionsWidget.buttonE.clicked.connect(self.pollE)
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.OptionsWidget.buttonA)
        options_layout.addWidget(self.OptionsWidget.buttonB)
        options_layout.addWidget(self.OptionsWidget.buttonC)
        options_layout.addWidget(self.OptionsWidget.buttonD)
        options_layout.addWidget(self.OptionsWidget.buttonE)
        self.OptionsWidget.setLayout(options_layout)

        # Reset and view results button VBox
        self.ResetWidget = QWidget(self)
        self.ResetWidget.reset_button = QPushButton('Reset')
        self.ResetWidget.reset_button.clicked.connect(self.reset_poll)
        self.ResetWidget.test_plot_button = QPushButton('Test Plot')
        self.ResetWidget.test_plot_button.clicked.connect(self.test_plot)
        self.ResetWidget.view_image_button = QPushButton('View Image')
        self.ResetWidget.view_image_button.clicked.connect(
            self.view_image_window)
        reset_layout = QVBoxLayout()
        reset_layout.addWidget(self.ResetWidget.reset_button)
        reset_layout.addWidget(self.ResetWidget.view_image_button)
        reset_layout.addWidget(self.ResetWidget.test_plot_button)
        self.ResetWidget.setLayout(reset_layout)

        # Organize MainGrid
        self.MainGrid = QGridLayout()
        # self.MainGrid.setSpacing(10)
        self.MainGrid.addWidget(self.OptionsWidget, 1, 0, 2, 1)
        self.MainGrid.addWidget(self.ResetWidget, 6, 0, 1, 1)
        self.MainGrid.addWidget(self.ResultsAxesWidget, 0, 1, 8, 11)

        # Set MainGrid as central widget of MainWindow
        self.MainWidget = QWidget(self)
        self.MainWidget.setLayout(self.MainGrid)
        self.setCentralWidget(self.MainWidget)

        # Initialize results and image axes
        self.results_ax = self.ResultsAxesWidget.figure.add_subplot(111)
        self.ResultsAxesWidget.figure.clear()
        self.image_ax = self.ViewImageWindow.ImageAxesWidget.figure.add_subplot(
            111)
        self.ViewImageWindow.ImageAxesWidget.figure.clear()

    def initModel(self):
        graph_path = 'frozen_inference_graph.pb'
        label_path = 'label_map.pbtxt'

        self.model = Model(graph_path, label_path)

    def initConn(self):
        self.connection = Connection(
            '10.42.0.1', 5001, '10.42.0.171', 5001)

    def initResults(self):
        self.options_list = ['A', 'B', 'C', 'D', 'E']
        self.results = Results()

    def poll_callback(self, option):
        tic = time.time()
        self.connection.send_cap_trigger()
        inc_data = self.connection.wait_image_data()
        image_np = inc_data
        toc = time.time()
        print('TOTAL TIME: {:.3f}'.format(toc-tic))
        tic = time.time()
        num_hands = self.model.detect(image_np)
        toc = time.time()
        print('INFERENCE TIME: {:.3f}'.format(toc-tic))

        self.results.add_result(option, num_hands)
        self.update_plot()

        self.show_results_image()

    def pollA(self):
        option = 'A'
        self.poll_callback(option)

    def pollB(self):
        option = 'B'
        self.poll_callback(option)

    def pollC(self):
        option = 'C'
        self.poll_callback(option)

    def pollD(self):
        option = 'D'
        self.poll_callback(option)

    def pollE(self):
        option = 'E'
        self.poll_callback(option)

    def reset_poll(self):
        self.results.reset_results()
        self.results_ax.clear()
        self.ResultsAxesWidget.figure.clear()
        self.ResultsAxesWidget.canvas.draw()

        self.image_ax.clear()
        self.ViewImageWindow.ImageAxesWidget.figure.clear()
        self.ViewImageWindow.ImageAxesWidget.canvas.draw()

    def update_plot(self):
        self.ResultsAxesWidget.figure.clear()

        # create an axis
        self.results_ax = self.ResultsAxesWidget.figure.add_subplot(111)

        results_dict = self.results.results_dict
        results = [results_dict[option] for option in results_dict]
        self.results_ax.bar(self.options_list, results, color='deepskyblue')

        rects = self.results_ax.patches
        labels = results
        for rect, label in zip(rects, labels):
            if label == 0:
                pass
            else:
                height = rect.get_height()
                self.results_ax.text(rect.get_x() + rect.get_width() / 2,
                                     height / 2, label, ha='center', va='top')

        # refresh canvas
        self.ResultsAxesWidget.canvas.draw()

    def test_plot(self):
        self.ResultsAxesWidget.figure.clear()

        # create an axis
        self.results_ax = self.ResultsAxesWidget.figure.add_subplot(111)

        results = [random.randint(0, 25) for _ in range(5)]
        self.results_ax.bar(self.options_list, results, color='deepskyblue')

        rects = self.results_ax.patches
        labels = results
        for rect, label in zip(rects, labels):
            if label == 0:
                pass
            else:
                height = rect.get_height()
                if height > 1:
                    self.results_ax.text(rect.get_x() + rect.get_width() / 2,
                                         height / 2, label, ha='center', va='top')
                else:
                    self.results_ax.text(rect.get_x() + rect.get_width() / 2,
                                         height / 2 + 2, label, ha='center', va='top')

        # Display test image
        dummy_image = Image.open('default_display_image.png')
        self.ViewImageWindow.ImageAxesWidget.figure.clear()
        self.image_ax = self.ViewImageWindow.ImageAxesWidget.figure.add_subplot(
            111)
        self.image_ax.imshow(dummy_image)
        self.image_ax.axis('off')

        # refresh canvases
        self.ResultsAxesWidget.canvas.draw()
        self.ViewImageWindow.ImageAxesWidget.canvas.draw()

    def show_results_image(self):
        if self.model.output_dict:
            vis_util.visualize_boxes_and_labels_on_image_array(
                self.model.image_np,
                self.model.output_dict['detection_boxes'],
                self.model.output_dict['detection_classes'],
                self.model.output_dict['detection_scores'],
                self.model.category_index,
                instance_masks=self.model.output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                min_score_thresh=self.model.detection_thresh,
                line_thickness=6)
            self.ViewImageWindow.ImageAxesWidget.figure.clear()
            self.image_ax = self.ViewImageWindow.ImageAxesWidget.figure.add_subplot(
                111)
            self.image_ax.imshow(self.model.image_np)
            self.image_ax.axis('off')
        else:
            # Display test image
            dummy_image = Image.open('default_display_image.png')
            self.ViewImageWindow.ImageAxesWidget.figure.clear()
            self.image_ax = self.ViewImageWindow.ImageAxesWidget.figure.add_subplot(
                111)
            self.image_ax.imshow(dummy_image)
            self.image_ax.axis('off')
        self.ViewImageWindow.ImageAxesWidget.canvas.draw()

    def view_image_window(self):
        self.ViewImageWindow.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = MainWindow()
    main.show()

    sys.exit(app.exec_())
