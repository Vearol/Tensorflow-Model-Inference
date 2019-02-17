import tensorflow as tf
import numpy
import sys
import os

path = sys.argv[1]
model_name = path.split('/')[-1]

loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(path + '.meta')
    loader.restore(sess, path)
    all_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for layer in all_layers:
        print('Saving layer: {0}\n'.format(layer.name))
        array = layer.eval()
           
        layer_dict = '../test/saved_Layers/' + layer.name.split('/')[0]
        layer_file = '../test/saved_Layers/' + layer.name.replace(':', '_') + '.npz'
        
        if not os.path.exists(layer_dict):
            os.makedirs(layer_dict)
            
        with open(layer_file, 'wb') as file1:
            numpy.savez(file1, array)