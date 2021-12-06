# braincell

An annotated demonstration of a "single-neuron neural network" model, as a teaching/learning tool for the fundamental underpinnings of machine learning and (in contemporary, loose usage) AI.

>Articulating the NN in 'atomic' unit-form highlights the lower-level statistical operations that are performed at the heart of machine learning: 

>In short,
>
> - "training" and "reasoning"/guessing in a given model,
> - through the adjustment of synaptic weights relative to known-outcome input data,
> - by iteratively calibrating those weights against a sigmoid function (and its derivative, as self-measured error)

Understanding of this foundational, mathematically-rooted process allows us to in turn better understand higher-level machine learning concepts of different flavours (be they associative, collocative, adversarial, etc.).

Our thingy:


## Usage
----

Depends on `numpy`.

Running `braincell` executes an end-to-end demonstration (initialisation, training, and two test cases).

```bash
$ python braincell.py
```

By default, `braincell` will output verbosely and perform 10,000 iterations (an accuracy:instantaneity sweet spot).

You can optionally supply (in any order) a specific integer-number of iterations, and/or `quiet` to turn off iteration verbosity:

```bash
$ python braincell.py 10 quiet

Random initial synaptic weights:
 [[0.80190343]
 [0.79406162]
 [0.89057025]]

Training for 10 iterations...

Post-training synaptic weights:
 [[ 1.70780518]
 [-0.71665902]
 [-0.23628915]]

Considering new situation: [1, 0, 0] -> ?
[0.84655139] for an expected output of 1.
Result inaccuracy was 15.3449%

Considering new situation: [0, 1, 0] -> ?
[0.32812911] for an expected output of 0.
Result inaccuracy was 32.8129%
```

You can also call and use the `SingleNeuronNetwork` class as you please. It can be used as-is for data with the same matrix configuration and value range as the example it is built around. 

However, for a use case with different data shape and value range, you will have to manually customise the matrix initialisation (as synaptic weights), matrix processing, and potentially how the sigmoid function normalises values. But, that amounts to taking the theory learned here, and applying it to a new problem space!
