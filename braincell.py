import sys
from collections import Counter

from numpy import exp, array, random, dot


def get_validated_inputs(sysargs):
    """Quick and dirty config parsing/validation."""
    
    # clean & coerce sys.argsv input
    args = []
    for arg in sysargs[1:]:  # omit first entry (it's just the filename)
        try:
            args.append(int(arg))
        except:
            args.append(arg)
    arg_types = tuple(map(type, args))
    if args and any((
        len(args) > 2,
        any(x > 1 for x in Counter(arg_types).values()),
        any(_type not in {int, str} for _type in set(arg_types)),
        any(arg != 'quiet' for arg in args if type(arg) is str),
    )):
        raise TypeError(
            "Invalid optional args received. Valid options are (in any order):"
            "\n    quiet -> turn iteration verbosity off"
            "\n    10000 -> or any other int, for # of trainign iterations"
        )

    num_iters = None
    verbosity = True

    # case-match ideal here, but don't lock to python 3.10+
    for arg in args:
        if   type(arg) is int:
            num_iters = arg
        else:  # value can only be 'quiet' given earlier validation
            verbosity = False

    return num_iters, verbosity


class SingleNeuronNetwork:

    # a visual divider for stdout
    DIVIDER = '----------------'

    def __init__(self, verbosity=True):
        """
        Initialise the 'neuron' / 'neural network'.
        """

        self.is_trained = False
        self.verbosity  = verbosity

        # get a random seed, but use the same one
        # for all initial synaptic weights
        random.seed(
            random.randint(0, 4294967296)  # high is 32-bit maxint (exclusive)
        )

        # model a single neuron with 3 inputs : 1 output
        # and randomize weights for a 3x1 matrix 
        #    (with values between -1 and 1, and mean of 0)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        print(f"\nRandom initial synaptic weights:\n {self.synaptic_weights}")

    def __sigmoid_func(self, x):
        """Functionally represent an S-shaped (sigmoid) curve."""

        # pass weighted sum of inputs through sigmoid function
        # in order to normalise them as values between 0 and 1
        return 1 / (1 + exp(-x))

    def __sigmoid_deriv(self, x):
        """
        Obtain the derivative of the sigmoid curve/function.

        Its derivative/gradient represents the level of 'confidence'
        in synaptic weights at a current given moment/iteration.
        """

        # compute gradient of curve,
        # i.e. given-moment confidence in synaptic weighting
        return x * (1 - x)

    def __measure_inaccuracy(self, expected, obtained):
        """Determine the percentage difference for a test case output guess."""
        return float(abs(min(expected, obtained) - max(expected, obtained)))

    def train(self, training_inputs, training_outputs, num_iters):
        """
        Train the NN via trial-and-error.
        Synaptic weights are adjusted with each iteration.
        """

        # default to 10k iterations if supplied count was 0 or None
        if not num_iters:
            num_iters = 10000
        print(f"\nTraining for {num_iters} iterations...")

        for i in range(num_iters):
            # pass train set through single-neuron NN
            output = self.reason_about(training_inputs)

            # compute 'error' amount (diff between actual vs. predicted output)
            err_rate = training_outputs - output

            # multiply error by input, then by gradient of Sigmoid curve, so that:
            # a) less confident synaptic weights become adjusted more
            # b) zero-inputs do not cause changes to weights
            adj = dot(
                training_inputs.T, # matrix.T is its orthogonal transposition
                err_rate * self.__sigmoid_deriv(output)
            )

            # adjust the synaptic weights nice n easy
            self.synaptic_weights += adj

            if self.verbosity:
                print(
                    f"\nIteration {i}:\n"
                    f"{str(output)}")

        self.is_trained = True
        print(f"\nPost-training synaptic weights:\n {self.synaptic_weights}")

    def reason_about(self, inputs):
        """
        Evaluate a given input based on current (trained) synaptic weights,
        by passing the dot product of inputs through the single-neuron NN.

        Fundamentally, this is what it means when an "AI" is "thinking",
        (or a machine learning model is "reasoning"), in a low-level sense.
        """
        return self.__sigmoid_func(dot(inputs, self.synaptic_weights))

    def encounter_input(self, test_case):
        """Pose a new (or specific) input case to the NN."""
        print(
            f"\nConsidering new situation: {str(test_case)} -> ?"
        )
        if not self.is_trained:
            print(
                f"WARNING: Your model is untrained; results will be random!"
                f"\nCall .train() on this instance with "
                f"an appropriately shaped array of training data."
            )
        result = self.reason_about(array(test_case))
        print(
            f"{result} for an expected output of {test_case[0]}."
            f"\nResult inaccuracy was "
            f"{self.__measure_inaccuracy(test_case[0], result):.4%}"
        )


if __name__ == "__main__":

    num_iters, verbosity = get_validated_inputs(sys.argv)

    # initialise single-neuron NN by calling its class
    nn = SingleNeuronNetwork(verbosity=verbosity)

    # create small training set: 3 inputs and 1 output in each case
    # IMPORTANT: the 'correct' output is simply the first element in each case!
    #            the NN will not "learn" this *as an axiomatic rule*,
    #            but rather constructs a strong statistical relationship
    #            between first element and output based on given training data.
    train_set_in = array(
        [
            [0, 1, 1],   # 0
            [1, 1, 1],   # 1
            [1, 0, 1],   # 1
            [0, 1, 1]    # 0
        ]
    )

    # the 'correct answers' for the input training 'puzzle' above, per row.
    # (as shown in the per-row comments above, i.e. always first element.)
    train_set_out = array(
        [
            [0, 1, 1, 0]
        ]
    ).T

    # run the training process with the training set a given number of times
    nn.train(train_set_in, train_set_out, num_iters)

    # test the trained NN instance with situations it hasn't seen yet:
    nn.encounter_input(
        [1, 0, 0]
    )
    nn.encounter_input(
        [0, 1, 0]
    )
