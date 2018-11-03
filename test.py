import io
import PIL.Image
import tensorflow as tf

full_path = '/home/dylan/Pictures/ee49_project_hands/images/944043026-612x612.jpg'
with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
encoded_jpg_io = io.BytesIO(encoded_jpg)
image = PIL.Image.open(encoded_jpg_io)
image = image.convert('L')

image.show()

