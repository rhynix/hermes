import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, sizes, learning_rate):
        self.sizes         = sizes
        self.learning_rate = learning_rate
        self.layers        = len(sizes)
        self.weights       = self.init_weights(sizes)

    def init_weights(self, sizes):
        stddev = 0.01
        mean   = 0

        return [stddev * np.random.randn(prev + 1, curr) + mean
                for prev, curr in zip(sizes, sizes[1:])]

    def learn(self, dataset, iterations):
        batch_size = 1000

        for i in range(iterations):
            self.learn_in_mini_batches(dataset, batch_size)
            results = self.feed_forward(dataset.inputs)[-1]

            yield self.cost(dataset.outputs, results)

    def learn_in_mini_batches(self, dataset, size):
        for idx in range(0, len(dataset), size):
            self.learn_batch(dataset[idx:idx+size])

    def learn_batch(self, dataset):
        examples = len(dataset)
        results  = self.feed_forward(dataset.inputs)
        changes  = self.changes(dataset.outputs, results)

        self.apply_changes(changes, examples)

    def feed_forward(self, inputs):
        results  = [inputs]

        for weights in self.weights:
            inputs          = self.add_bias(results[-1])
            weighted_inputs = np.dot(inputs, weights)

            results.append(scipy.special.expit(weighted_inputs))

        return results

    def cost(self, outputs, results):
        examples = len(results)

        return np.sum(np.square(outputs - results)) / (2 * examples)

    def changes(self, outputs, results):
        changes = [np.zeros(weights.shape) for weights in self.weights]

        delta       = (results[-1] - outputs) * (1 - results[-1]) * results[-1]
        changes[-1] = np.dot(self.add_bias(results[-2]).transpose(), delta)

        for i in range(2, self.layers):
            weights       = self.weights[-i + 1]
            weighted_errs = np.dot(delta, weights[1:].transpose())

            delta             = weighted_errs * (1 - results[-i]) * results[-i]
            results_with_bias = self.add_bias(results[-i-1])
            changes[-i]       = np.dot(results_with_bias.transpose(), delta)

        return changes

    def apply_changes(self, changes, examples):
        self.weights = [weights - (self.learning_rate / examples) * changes
                        for weights, changes in zip(self.weights, changes)]

    def add_bias(self, results):
        return np.hstack((np.ones((len(results), 1)), results))

    def accuracy(self, dataset):
        results = self.feed_forward(dataset.inputs)[-1]

        output_classes = np.argmax(dataset.outputs, axis=1)
        result_classes = np.argmax(results,         axis=1)

        correct = np.sum(output_classes == result_classes)
        total   = len(dataset)

        return correct / total
