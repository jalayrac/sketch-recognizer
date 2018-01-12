#!/usr/bin/env python
import numpy as np
import os
import sys
# EDIT HERE: specify your caffe location.
caffe_root = "/home/jalayrac/src/caffe"
sys.path.insert(0, caffe_root+'/python')
import caffe
import time


PRETRAINED_FILE = './models/finetune_googledraw_iter_360000.caffemodel'
sketch_pred_model = './models/googlenet_deploy.prototxt'

# Set CPU/GPU mode (replace by caffe.set_mode_gpu() for GPU usage).
caffe.set_mode_cpu()
sketch_net_pred = caffe.Net(sketch_pred_model, PRETRAINED_FILE, caffe.TEST)

output_layer_pred = 'loss3/classifier_s'

transformer = caffe.io.Transformer({'data': np.shape(sketch_net_pred.blobs['data'].data)})
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_transpose('data',(2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
file_list_class = './models/class_list.txt'

class_list = []

with open(file_list_class, 'r') as f:
    for line in f:
        class_list.append(line.rstrip())


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict_sketch_category(path_img):
    """Predicts category of a sketch.

    Prints the top 5 prediction with their probability score.
    
    Args:
        path_img: Path to the image containing the sketch (png, jpg).
    """            
    
    start_time = time.time()
    
    # Load the image and preprocess it.
    sketch_in = (transformer.preprocess('data', caffe.io.load_image(path_img)))
    sketch_in = np.reshape([sketch_in], np.shape(sketch_net_pred.blobs['data'].data))

    # Forward pass in the network.
    query = sketch_net_pred.forward(data=sketch_in)
    query = np.copy(query[output_layer_pred])
    X_query = np.array(query.ravel())

    # Convert to probability with softmax function.
    probabilities = softmax(X_query)

    # Display the predictions.
    top_indexes = np.argsort(-X_query)
    duration_time = time.time()-start_time
    print("%s: %s (%0.3f), %s (%0.2f), %s (%0.2f), %s (%0.2f), %s (%0.2f) (served in %0.3f s)" %
          (os.path.basename(path_img),
           class_list[top_indexes[0]], probabilities[top_indexes[0]],
           class_list[top_indexes[1]], probabilities[top_indexes[1]],
           class_list[top_indexes[2]], probabilities[top_indexes[2]],
           class_list[top_indexes[3]], probabilities[top_indexes[3]],
           class_list[top_indexes[4]], probabilities[top_indexes[4]],
           duration_time))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        predict_sketch_category(path)
