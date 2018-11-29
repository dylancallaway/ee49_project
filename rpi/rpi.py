#!/home/pi/ee49_project/rpi/venv/bin/python

import socket
import pickle
import time
import picamera
import io

stream = io.BytesIO()

cam = picamera.PiCamera(resolution=(2592, 1944))

cam.capture(stream, format='jpeg')

stream.seek(0)
