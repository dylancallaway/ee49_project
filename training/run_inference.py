import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import socket
import pickle
from PIL import Image

import time


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

        # Receiving images sockets setup
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_sock.bind((self.recv_host, self.recv_port))
        print('Listening at: {}:{}'.format(
            self.recv_host, str(self.recv_port)))

    def wait_data(self):
        self.recv_sock.listen(1)
        data = b''
        self.conn, self.addr = self.recv_sock.accept()
        print('Incoming connection from:', self.addr)
        while True:
            inc_data = self.conn.recv(1024)
            if inc_data == b'':
                print('Received {} bytes.'.format(len(data)))
                break
            else:
                data += inc_data
        return data

    def send_results(self, results):
        # Send results socket setup
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_sock.connect((self.send_host, self.send_port))
        self.send_sock.sendall(results)
        self.send_sock.close()


if __name__ == '__main__':

    graph_path = '/home/dylan/ee49_project/training/models/faster_rcnn_resnet50_lowproposals_coco_2018_01_28/frozen_inference_graph.pb'
    label_path = '/home/dylan/ee49_project/training/data/tf_records/hands/label_map.pbtxt'
    model = Model(graph_path, label_path)

    image_path = 'training/test_images/first-gen-hand-raise-uc-davis.jpg'
    image_pil = Image.open('../' + image_path)
    image_np = np.array(image_pil)

    tic = time.time()
    detected_hands = model.detect(image_np)
    toc = time.time()
    print('ELAPSED TIME: {:.3f}'.format(toc-tic))

    model.display_results()
