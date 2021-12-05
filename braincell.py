import sys
from numpy import exp, array, random, dot


class SingleNeuronNetwork:

    def __init__(self, seed=0):
        """
        Initialise the 'neuron' / 'neural network'.
        """

        # use the same rng seed every run
        random.seed(seed)

        # model a single neuron with 3 inputs : 1 output
        # and randomize weights for a 3x1 matrix 
        #    (with values between -1 and 1, and mean of 0)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid_func(self, x):
        """Functionally represent an S-shaped (sigmoid) curve."""

        # pass weighted sum of inputs through sigmoid function
        # in order to normalise them as values between 0 and 1
        return 1 / (1 + exp(-x))

    def __sigmoid_deriv(self, x):
        """Obtain the derivative of the sigmoid curve/function."""

        # compute gradient of curve,
        # i.e. given-moment confidence in synaptic weighting
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, num_iters):
        """
        Train the NN via trial-and-error.
        Synaptic weights are adjusted with each iteration.
        """

        for i in range(num_iters):
            # pass train set through single-neuron NN
            output = self.reason_about(training_inputs)

            # compute 'error' amount (diff between actual vs. predicted output)
            err_rate = training_outputs - output

            # multiply error by input, then by gradient of Sigmoid curve, so that:
            # a) less confident synaptic weights become adjusted more
            # b) zero-inputs do not cause changes to weights
            adj = dot(
                training_inputs.T, # calling matrix.T represents a matrix orthogonally
                err_rate * self.__sigmoid_deriv(output)
            )

            # adjust the synaptic weights nice n easy
            self.synaptic_weights += adj

            print(
                f"Iteration {i}:\n"
                f"{str(output)}")

    def reason_about(self, inputs):
        """
        Evaluate a given input based on current (trained) synaptic weights.

        Fundamentally, this is what it means when an "AI" is "thinking",
        in a low-level sense.
        """


        # pass (dot product) inputs through the single-neuron NN
        return self.__sigmoid_func(dot(inputs, self.synaptic_weights))

    def encounter_new(self, test_case):
        """Pose a new (or specific) input case to the NN."""

        print(
            f"\nConsidering new situation {str(test_case)} -> ?:\n"
            f"{self.reason_about(array(test_case))}"
        )


if __name__ == "__main__":

    # initialise single-neuron NN by calling its class
    nn = SingleNeuronNetwork()
    print(f"Random initial synaptic weights: {nn.synaptic_weights}")

    # create small training set: 3 inputs and 1 output in each case
    train_set_in = array(
        [
            [0, 1, 1],   # 0
            [1, 1, 1],   # 1
            [1, 0, 1],   # 1
            [0, 1, 1]    # 0
        ]
    )

    train_set_out = array(
        [[0, 1, 1, 0]]
    ).T

    # run the training process with the training set a given number of times
    nn.train(train_set_in, train_set_out, 10000)
    print(f"\nPost-training synaptic weights:\n {nn.synaptic_weights}")

    # test the trained NN instance with a new input situation:
    nn.encounter_new(
        [1, 0, 0]
    )
