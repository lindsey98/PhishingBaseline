import numpy as np
from matplotlib.pyplot import imread
from skimage.transform import rescale, resize
import os
from tqdm import tqdm
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Subtract, Reshape
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Input, Lambda, GlobalMaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.applications.vgg16 import VGG16


# Compute L2 distance between embeddings
def compute_distance_pair(layer1, layer2):
    diff = layer1 - layer2
    l2_diff = np.mean(diff ** 2)
    return l2_diff


# Compute distance between the query set and all training examples
def compute_all_distances(query_set, train_set):
    train_size = train_set.shape[0]
    pairwise_distance = np.zeros([query_set.shape[0], train_size])
    for i in range(0, query_set.shape[0]):
        pair1 = query_set[i, :]
        for j in range(0, train_size):
            pair2 = train_set[j, :]
            l2_diff = compute_distance_pair(pair1, pair2)
            pairwise_distance[i, j] = l2_diff
    return pairwise_distance


# Sample a batch of triplets for model training
def get_batch(batch_size, X_train, y_train, new_label_dict=None):
    if new_label_dict is None:  # construct label dictionary myself
        new_label_dict = {}

        for i in range(len(y_train)):
            if y_train[i] not in new_label_dict.keys():
                new_label_dict[y_train[i]] = []
            new_label_dict[y_train[i]].append(i)

    h = X_train.shape[1]
    w = X_train.shape[2]
    c = X_train.shape[3]
    triple = [np.zeros((batch_size, h, w, c)) for i in range(3)]

    for i in range(batch_size):
        target = np.random.choice(list(new_label_dict.keys()), 1)[0]
        anchor_ind = np.random.choice(new_label_dict[target], 1)[0]
        pos_ind = np.random.choice(new_label_dict[target], 1)[0]

        target_neg = np.random.choice(list(new_label_dict.keys()), 1)[0]
        while target_neg == target:
            target_neg = np.random.choice(list(new_label_dict.keys()), 1)[0]
        neg_ind = np.random.choice(new_label_dict[target_neg], 1)[0]

        triple[0][i, :, :, :] = X_train[anchor_ind, :]
        triple[1][i, :, :, :] = X_train[pos_ind, :]
        triple[2][i, :, :, :] = X_train[neg_ind, :]

    return triple


# Model architecture
def define_triplet_network(input_shape, new_conv_params):
    # Input_shape: shape of input images
    # new_conv_params: dimension of the new convolution layer [spatial1,spatial2,channels]

    # Define the tensors for the three input images
    anchor_input = Input(input_shape)
    positive_input = Input(input_shape)
    negative_input = Input(input_shape)

    # Use VGG as a base model
    base_model = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)

    x = base_model.output
    x = Conv2D(new_conv_params[2], (new_conv_params[0], new_conv_params[1]), activation='relu',
               kernel_initializer='he_normal', kernel_regularizer=l2(2e-4))(x)
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)

    # Generate the encodings (feature vectors) for the two images
    encoded_a = model(anchor_input)
    encoded_p = model(positive_input)
    encoded_n = model(negative_input)

    mean_layer = Lambda(lambda x: K.mean(x, axis=1))

    square_diff_layer = Lambda(lambda tensors: K.square(tensors[0] - tensors[1]))
    square_diff_pos = square_diff_layer([encoded_a, encoded_p])
    square_diff_neg = square_diff_layer([encoded_a, encoded_n])

    square_diff_pos_l2 = mean_layer(square_diff_pos)
    square_diff_neg_l2 = mean_layer(square_diff_neg)

    # Add a diff layer
    diff = Subtract()([square_diff_pos_l2, square_diff_neg_l2])
    diff = Reshape((1,))(diff)

    # Connect the inputs with the outputs
    triplet_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=diff)

    # return the model
    return triplet_net


# Define triplet loss
def custom_loss(margin):
    def loss(y_true, y_pred):
        loss_value = K.maximum(y_true, margin + y_pred)
        loss_value = K.mean(loss_value, axis=0)
        return loss_value

    return loss