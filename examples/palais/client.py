#!/usr/bin/env python
# coding: utf-8
import json
import socket
import base64

hote = "localhost"
port = 4004

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect((hote, port))
print "Connection on {}".format(port)

# Open the image and encoded it in a string.
with open('../../samples/sample_1.png', 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read())

first_phase_data = {'query_type': 1,
                    'image': encoded_string}

query = json.dumps(first_phase_data, ensure_ascii=False)
socket.send(query)
answer = socket.recv(port)
data_back = json.loads(answer)

print data_back['name_class']
print data_back['confidence']
socket.close()
