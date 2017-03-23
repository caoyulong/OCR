import csv
import numpy as np
from numpy import matrix
from collections import namedtuple
import math
import random
import json
import os

class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    FILE_PATH = 'nn.json'

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes, data_matrix, data_labels, training_indices, train_num):

        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        self.sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)
        self.data_matrix = data_matrix
        self.data_labels = data_labels

        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes

        # Step 1: Initialize weights to small numbers
        self.theta1 = self._rand_initialize_weights(num_input_nodes, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, num_output_nodes)
        self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(1, num_output_nodes)

        TrainData = namedtuple('TrainData', ['y0', 'label'])
        # load ann
        if os.path.isfile(self.FILE_PATH):
            self.load()
            return
        # Train using sample data
        if train_num < 1:
            train_num = 1
        if train_num > 10:
            train_num = 10
        for j in range(train_num):
            self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    # The sigmoid activation function. Operates on scalars.
    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)

    def _sigmoid_prime_scalar(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def train(self, training_data_array):
        for data in training_data_array:
            # Step 2: Forward propagation
            y1 = np.dot(np.mat(self.theta1), np.mat(data.y0).T)
            sum1 = np.add(y1, np.mat(self.input_layer_bias)) # Add the bias
            y1 = self.sigmoid(sum1)

            y2 = np.dot(np.array(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias) # Add the bias
            y2 = self.sigmoid(y2)

            # Step 3: Back propagation
            actual_vals = [0 for i in range(self.num_output_nodes)] # actual_vals is a python list for easy initialization and is later turned into an np matrix (2 lines down).
            actual_vals[data.label] = 1
            output_errors = np.mat(actual_vals).T - np.mat(y2)
            hidden_errors = np.multiply(np.dot(np.mat(self.theta2).T, output_errors), self.sigmoid_prime(sum1))

            # Step 4: Update weights
            self.theta1 += self.LEARNING_RATE * np.dot(np.mat(hidden_errors), np.mat(data.y0))
            self.theta2 += self.LEARNING_RATE * np.dot(np.mat(output_errors), np.mat(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):

        y1 = np.dot(np.mat(self.theta1), np.mat(test).T)
        y1 = np.add(y1, np.mat(self.input_layer_bias)) # Add the bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias) # Add the bias
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        json_nn = {
            "theta1":[row.tolist()[0] for row in self.theta1],
            "theta2":[row.tolist()[0] for row in self.theta2],
            "b1":self.input_layer_bias.tolist(),
            "b2":self.hidden_layer_bias.tolist()
        }
        with open(self.FILE_PATH, 'w') as nnfile:
            json.dump(json_nn, nnfile)

    def load(self):
        print 'load file nn.json'
        with open(self.FILE_PATH) as nnfile:
            nn = json.load(nnfile)
        self.theta1 = np.mat(nn['theta1'])
        self.theta2 = np.mat(nn['theta2'])
        self.input_layer_bias = np.array(nn['b1'])
        self.hidden_layer_bias = np.array(nn['b2'])