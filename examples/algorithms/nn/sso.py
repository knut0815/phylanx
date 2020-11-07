#  Copyright (c) 2020 N. Wu
#  Copyright (c) 2020 P. Diehl
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from phylanx import Phylanx,PhylanxSession

PhylanxSession.debug = False
PhylanxSession.disable_decorator = False

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import copy
import time
#from common import Times
#from matrix_free import matrix_free_operator
#from gcr import gcr
import argparse
#import nn
from nn import *
#import nn_state
from nn_state import *
#import read_write
from read_write import *
from bp import *
from tester import *

#np.seterr(invalid='ignore')

#@Phylanx
def get_args():
    parser = argparse.ArgumentParser(description='SSO')
    parser.add_argument('--version', type=int, default=0,
                        help='algorithm version to run')
    parser.add_argument('--train_label_dir',
                        default='../data/train-labels-idx1-ubyte',
                        help='directory of training label')
    parser.add_argument('--train_image_dir',
                        default='../data/train-images-idx3-ubyte',
                        help='directory of training image')
    parser.add_argument('--test_label_dir',
                        default='../data/t10k-labels-idx1-ubyte',
                        help='directory of testing label')
    parser.add_argument('--test_image_dir',
                        default='../data/t10k-images-idx3-ubyte',
                        help='directory of testing image')
    parser.add_argument('-n', type=int, default=8192,
                        help='number of training images')
    parser.add_argument('-i', type=int, default=20,
                        help='max number of epochs')
    parser.add_argument('--batch', type=int, default=1,
                        help='batch size')
    parser.add_argument('--hidden', type=int, default=15,
                        help='size of hidden layer')
    #parser.add_argument('--csv_train', required=True,
    #                    help='training data file in csv format')
    #parser.add_argument('--csv_test', required=True,
    #                    help='test data file in csv format')
    parser.add_argument('--alpha', type=float, default=0.3333,
                        help='training update weight')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='sigmoid stretching weight')
    parser.add_argument('--max_krylov', type=int, default=10,
                        help='how many krylov iterations to run')
    parser.add_argument('--max_newton', type=int, default=10,
                        help='how many newton iterations to run')
    parser.add_argument('--max_beta', type=int, default=10,
                        help='how many beta steps to run')
    parser.add_argument('--newton_tol', type=float, default=1.E-3,
                        help='tolerance for Newton iteration convergence')
    parser.add_argument('--krylov_tol', type=float, default=1.E-3,
                        help='tolerance for krylov iteration convergence')
    parser.add_argument('--acc_dir',
                        default='../acc_dir/accuracy.json',
                        required=False,
                        help='directory to store the test accuracy')
    parser.add_argument('--read_nn',
                        help='serialize nn to directory after training')
    parser.add_argument('--write_nn',
                        help='deserialize nn from directory before training')
    return parser.parse_args()


#@Phylanx(debug=True)
def read_train(train_image_dir, train_label_dir, max_num_images):
    """
    Read traning set:Each image has Intensity from 0 to 255, and also need to
    converts a class vector (integers) to binary class matrix, which is similar
    to_categorical function.
    Return training images and training labels
    """
    train_images_whole = idx2numpy.convert_from_file(train_image_dir)
    train_labels_whole = idx2numpy.convert_from_file(train_label_dir)
    num_images = min(len(train_images_whole), max_num_images)
    train_images = train_images_whole[:num_images]
    train_labels = train_labels_whole[:num_images]
    train_images = train_images / 255
    train_labels = np.eye(10)[train_labels]
    image_size = train_images[0].shape[0] * train_images[0].shape[1]
    train_images = train_images.reshape(train_images.shape[0], image_size, 1)
    return train_images, train_labels

#@Phylanx
def read_test(test_image_dir, test_label_dir):
    """
    Read test set: Each image has Intensity from 0 to 255, and also need to
    converts a class vector (integers) to binary class matrix, which is similar
    to_categorical function.
    Return test images and test labels
    """
    test_images = idx2numpy.convert_from_file(test_image_dir)
    test_labels = idx2numpy.convert_from_file(test_label_dir)
    test_images = test_images / 255
    test_labels = np.eye(10)[test_labels]
    image_size = test_images[0].shape[0] * test_images[0].shape[1]
    test_images = test_images.reshape(test_images.shape[0], image_size, 1)
    return test_images, test_labels

@Phylanx
def train(id, A, dict_nn_state, batch_sz, train_images, train_labels, test_images,
         test_labels, alpha, max_iterations):
    test_acc = 0
    t_total = 0
    if id == 0:
        data = bp_cpu(A,dict_nn_state, train_images, train_labels, alpha,
                                    max_iterations, test_images, test_labels,
                                    test_cpu_2)
        test_acc = data[0]
        t_total = data[1]
        
    elif id ==1:
        data = batch_bp_cpu(A,dict_nn_state, batch_sz, train_images, train_labels, alpha,
                                    max_iterations, test_images, test_labels,
                                    test_cpu_2)
        
        test_acc = data[0]
        t_total = data[1]

    return [test_acc, t_total]


if __name__ == "__main__":
    args = get_args()

   

    n = args.n
    train_image_dir = args.train_image_dir
    train_label_dir =  args.train_label_dir
    
    train_images, train_labels = read_train(train_image_dir,
                                            train_label_dir, n)
    
    test_image_dir = args.test_image_dir
    test_label_dir = args.test_label_dir
    
    test_images, test_labels = read_test(test_image_dir,
                                         test_label_dir)
    
  
    beta = args.beta
    hidden = int(args.hidden)
    
    image_shape = np.shape(train_images[0])
    input_layer = image_shape[0]
    layers = [input_layer, hidden, 10]
    print("here 1")
    dict_nn = init_nn(dict_nn,layers)
    print("here")
    init_A(dict_nn, args.read_nn)
    write_A(dict_nn, args.write_nn)

   
    alpha = args.alpha
    i =  args.i
    batch = args.batch
    version = args.version
   
    data42 = train(version, dict_nn, dict_nn, batch, train_images, train_labels,
                test_images, test_labels, alpha, i)

    test_acc = data42[0]
    t_total = data42[1]


    for epoch in range(0, args.i):
        #print("test_acc", test_acc)
        print('epoch:{}, version: {}, batch_size: {}, test accu: {}, total time:{}.'.format(epoch, version, batch, test_acc[epoch], t_total[epoch]))


    #save test accuracy for plot9
    #acc_dir = args.acc_dir
    #results = {'test_acc': test_acc, 'time': args.i}
    #with open(acc_dir, 'w') as fp:
    #    json.dump(results, fp)
