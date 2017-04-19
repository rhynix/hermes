#!/usr/bin/env python3

import csv
import argparse
import numpy as np

from neural_network import NeuralNetwork
from collections import namedtuple

Dataset = namedtuple('Dataset', ['inputs', 'outputs'])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--file', required=True, type=str,
        help='The path to CSV file containing the MNIST dataset'
    )

    parser.add_argument(
        '-r', '--learning-rate', default=0.3, type=float,
        help='The learning rate'
    )

    parser.add_argument(
        '-l', '--layer-size', default=[30], type=int, nargs='*',
        help='The sizes of the hidden layers of the network'
    )

    parser.add_argument(
        '-t', '--training-size', default=50000, type=int,
        help='''
        The size of the dataset used for training. The rest of the dataset \
        will be used for evaluation
        '''
    )

    parser.add_argument(
        '-i', '--iterations', default=100, type=int,
        help='The number of times to process the full dataset'
    )

    return vars(parser.parse_args())

def output_for_class(output_class, size):
    return [output_class == idx for idx in range(size)]

def read_dataset(file_name, output_size):
    inputs  = [];
    outputs = [];

    with open(file_name, 'r') as handle:
        for row in csv.reader(handle):
            example_data    = list(map(int, row))
            example_inputs  = example_data[1:]
            example_outputs = output_for_class(example_data[0], output_size)

            inputs.append(example_inputs)
            outputs.append(example_outputs)

    return Dataset(np.array(inputs), np.array(outputs))

def split_dataset(dataset, size):
    first_dataset  = Dataset(dataset.inputs[:size], dataset.outputs[:size])
    second_dataset = Dataset(dataset.inputs[size:], dataset.outputs[size:])

    return (first_dataset, second_dataset)

def report_initial_cost(nn, dataset):
    results = nn.feed_forward(dataset.inputs)
    cost    = nn.cost(dataset.outputs, results[-1])

    print('Initial cost: %.10f' % (cost,))

def learn_and_report_cost(nn, dataset, iterations):
    learn = nn.learn(dataset.inputs, dataset.outputs, iterations)

    for index, cost in enumerate(learn):
        print('Cost(%04i): %.10f' % (index + 1, cost))

def report_evaluation(nn, dataset):
    accuracy = nn.evaluate(dataset.inputs, dataset.outputs) * 100
    print('Accuracy: %.3f%%' % (accuracy,))

def main():
    args = parse_args()

    hidden_layer_sizes = args['layer_size']
    iterations         = args['iterations']
    training_size      = args['training_size']
    file_name          = args['file']
    learning_rate      = args['learning_rate']

    layer_sizes = [28 * 28, *hidden_layer_sizes, 10]

    print('Reading dataset...')

    dataset                      = read_dataset(file_name, layer_sizes[-1])
    training_set, evaluation_set = split_dataset(dataset, training_size)

    nn = NeuralNetwork(layer_sizes, learning_rate)

    report_initial_cost(nn, training_set)
    learn_and_report_cost(nn, training_set, iterations)
    report_evaluation(nn, evaluation_set)

if __name__ == '__main__':
    main()
