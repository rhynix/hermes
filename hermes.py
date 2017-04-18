#!/usr/bin/env python3

import csv
import argparse
import numpy as np
from neural_network import NeuralNetwork

def read_dataset(file_name, sizes):
    output_size = sizes[-1]
    inputs      = [];
    outputs     = [];

    with open(file_name, 'r') as handle:
        for row in csv.reader(handle):
            data         = [int(x) for x in row]
            output_class = data[0]

            outputs.append([output_class == idx for idx in range(output_size)])
            inputs.append(data[1:])

    return (np.array(inputs), np.array(outputs))

def arguments():
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

def main():
    args = arguments()

    hidden_layer_sizes = args['layer_size']
    iterations         = args['iterations']
    training_size      = args['training_size']
    file_name          = args['file']
    learning_rate      = args['learning_rate']

    layer_sizes = [28 * 28, *hidden_layer_sizes, 10]

    print('Reading dataset...')

    nn              = NeuralNetwork(layer_sizes, learning_rate)
    inputs, outputs = read_dataset(file_name, layer_sizes)

    print('Starting training...')

    training_inputs = inputs[:training_size]
    eval_inputs     = inputs[training_size:]

    training_outputs = outputs[:training_size]
    eval_outputs     = outputs[training_size:]

    results      = nn.feed_forward(training_inputs)
    initial_cost = nn.cost(training_outputs, results[-1])

    print('Initial cost: %.10f' % (initial_cost,))

    learn = nn.learn(training_inputs, training_outputs, iterations)

    for index, cost in enumerate(learn):
        print('Cost(%05i): %.10f' % (index + 1, cost))

    accuracy = nn.evaluate(eval_inputs, eval_outputs) * 100
    print('Accuracy: %.3f%%' % (accuracy,))

if __name__ == '__main__':
    main()
