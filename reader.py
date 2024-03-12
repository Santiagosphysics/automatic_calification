# from __feature__ import absolute_import, fivision, print_function, unicode_literals
import tensorflow as tf 
import tensorflow_datasets as tfds 

import math 
import numpy as np 
import matplotlib.pyplot as plt
import logging 

logger = tf.get_logger()

logger.setlevel(logging.ERROR)

dataset, metadata = tfds.load('mnist', as_suppervised = True, with_info = True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = [
    'Cero', 'Uno', 'Dos','tres','Cuatro','Cinco','Seis','Siete','Ocho','Nueve'
]

num_train_examples = metadata.splits['train'].num_examples 
num_test_examples = metadata.splits['test'].num_examples 

# Normalizar, de cero a uno 
def normalize


çç