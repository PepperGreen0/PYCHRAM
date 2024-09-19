import numpy as np

from neuralnetwork import NeuralNetwork

if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork(2)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    # training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[1, 5],
                                [1, 4],
                                [2, 4],
                                [5, 3],
                                [4, 2],
                                [5, 2],])

    training_outputs = np.array([[0, 0, 0, 1, 1, 1]]).T

    # training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    #user_input_three = str(input("User Input Three: "))

    print("Considering New Situation: ", user_input_one, user_input_two)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two])))
    print("Wow, we did it!")
