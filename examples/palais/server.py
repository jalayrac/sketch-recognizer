#!/usr/bin/env python
# coding: utf-8

import json
import socket
import time
import numpy as np
import sys
from cStringIO import StringIO
from PIL import ImageFile
# EDIT: replace with your caffe location.
caffe_root = "/home/jalayrac/src/caffe"
sys.path.insert(0, caffe_root+'/python')
import caffe

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up caffe (EDIT here if you want gpu usage: replace by caffe.set_mode_gpu()
caffe.set_mode_cpu()

# Preloading of the pictionnary model.
pictionnary_model_file = '../../models/finetune_googledraw_iter_360000.caffemodel'
pictionnary_definition_file = '../../models/googlenet_deploy.prototxt'
pictionnary_model = caffe.Net(pictionnary_definition_file, pictionnary_model_file, caffe.TEST)

# Defining the output layer.
pictionnary_output_layer = 'loss3/classifier_s'

# Getting the class list.
file_list_class = '../../models/class_list.txt'

class_list = []
with open(file_list_class, 'r') as f:
    for line in f:
        class_list.append(line.rstrip())

# Set the transformer.
transformer = caffe.io.Transformer({'data': np.shape(pictionnary_model.blobs['data'].data)})
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def recv_timeout(socket_client, timeout=2):
    socket_client.setblocking(0)
    total_data = []
    begin = time.time()
    while 1:
        if total_data and time.time()-begin > timeout:
            break

        elif time.time()-begin > timeout*2:
            break

        try:
            data = socket_client.recv(8192)

            if data:
                total_data.append(data)
                begin = time.time()
            else:
                time.sleep(0.1)
        except:
            pass
    return ''.join(total_data)


socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind(('', 4004))

while True:
    socket.listen(1)
    client, address = socket.accept()
    print "{} connected".format(address)
    query = recv_timeout(client)
    json_data = json.loads(query)
    print json_data['query_type']

    if json_data['query_type'] == 1:

        # Pictionnary mode.
        file_pngdata = StringIO(json_data['image'].decode('base64'))
        image = caffe.io.load_image(file_pngdata)

        # Process the image trough the network.
        sketch_in = (transformer.preprocess('data', image))
        sketch_in = np.reshape([sketch_in],
                               np.shape(pictionnary_model.blobs['data'].data))

        query = pictionnary_model.forward(data=sketch_in)
        query = np.copy(query[pictionnary_output_layer])

        X_query = np.array(query.ravel())
        probabilities = softmax(X_query)
        top_indexes = np.argsort(-X_query)

        # Send the response.
        classes_out = [class_list[top_indexes[i]] for i in range(5)]
        confidence_score = probabilities[top_indexes[:5]]
        response_data = {'query_type': 1,
                         'confidence': confidence_score.tolist(),
                         'name_class': classes_out}

        response = json.dumps(response_data, ensure_ascii=False)
        client.send(response)

print "Close"

client.close()
socket.close()
