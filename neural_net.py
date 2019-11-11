from numpy import exp, array, dot
import pandas as pd

import fetch
from fetch import normalized


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # sigmoid function for the s curve.
    #passing values through the functions to make sure
    #they lie between 0-1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # calculating the derivative of the curve which acts as the gradient
    #which shows the level of confidence
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # training neural net through trial and error changing values in each iteration
    def train(self, training_inputs, training_outputs, number_of_iterations):
        for iteration in range(number_of_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_inputs)

            # calculating difference between desired output and layer 2
            layer2_error = training_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1 += layer1_adjustment
            self.layer2 += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("Layer 1 (2 neurons,  3 inputs): ")
        print(self.layer1)
        print("Layer 2 (1 neuron, 2 inputs):")
        print(self.layer2)

if __name__ == "__main__":
    # Create layer 1 (2 neurons, each with 3 inputs)
    layer1 = array([[0.2, 0.1], [0.3, 0.1], [0.2, 0.1]])

    # Create layer 2 (a single neuron with 2 inputs)
    layer2 = array([[0.5, 0.1]]).T

    # create an input list

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    # print ("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    normalized_set = normalized()
    # The training set. We have 6 examples, each consisting of 3 input values
    # and 1 output value.
    print(normalized_set['input1'][0])

    training_inputs = array(
        [
            [normalized_set['input1'][0], normalized_set['input2'][0], normalized_set['input3'][0]],
            [normalized_set['input1'][1], normalized_set['input2'][1], normalized_set['input3'][1]],
            [normalized_set['input1'][2], normalized_set['input2'][2], normalized_set['input3'][2]],
            [normalized_set['input1'][3], normalized_set['input2'][3], normalized_set['input3'][3]],
            [normalized_set['input1'][4], normalized_set['input2'][4], normalized_set['input3'][4]],
            [normalized_set['input1'][5], normalized_set['input2'][5], normalized_set['input3'][5]]
        ])

    training_outputs = array(
        [[
            normalized_set['output'][0],
            normalized_set['output'][1],
            normalized_set['output'][2],
            normalized_set['output'][3],
            normalized_set['output'][4],
            normalized_set['output'][5]
        ]]).T

    print("Training set: ", training_inputs)
    print("Training set: ", training_outputs)

    # training the neural network 90k times before making adjustments
    neural_network.train(training_inputs, training_outputs, 90000)

    print("Weights after training is done: ")
    neural_network.print_weights()

    # testing the NN with new weights
    output = neural_network.think(array([0.5, 0.6, 0.1]))
    print("The new weights: ", output[0])
    print("The expected output : ", output[1])