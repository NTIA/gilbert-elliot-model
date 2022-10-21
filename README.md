# gilbert-elliot-model

The Gilbert-Elliot burst error model is a popular and effective tool for treating bursty (non-independent) errors in communication links.
This software accompanies the NTIA Technical Memorandum 23-XXXX.
It provides functions for simulating errors using the model and analyzing error arrays to estimate model parameters.
It can also make an estimate of the most likely version of the model to emit a given observation: the two-parameter $(p, r)$, three-parameter $(p, r, h)$ or four-parameter $(p, r, k, h)$.

Upon installation the package provides a command line interface which allows users to determine valid model control values and simulate errors.

## Installation
Navigate to the directory containing the source code and run:

```
pip install .
```

## Documentation
The software includes doc strings for all methods, viewable via `help(method)`.
For example to see the help for the `simulate` method:
```
import gilbert_elliot_model as ge
print(help(ge.simulate))
```

Here we list the primary methods included in the package with a brief description of each 
* `simulate` - Simulate errors using the Gilbert-Elliot burst model and a variety of controls as parameters.
* `determine_model` - Determine most likely version of Gilbert-Elliot model given observation.
* `model_error_statistics` - Determine error statistics given model parameters.
* `data_error_statistics` - Measure error statsitics from error pattern.
* `fit_hmm` - Fit two-state Hidden Markov model to data in observation.
* `valid_regions` - Determine valid ranges for Gilbert-Elliot model controls and error statistics.
* `valid_parameters` - Return valid parameters either as a list of pairs or a set of all parameters.

### Selecting a model

It is worth noting that more than one set of model parameters could yield the same or very similar error patterns.
Thus, unless there is a good reason for selecting a more complicated model (such as the three- or four-parameter) models, we suggest using the two-parameter model.
More complicated models are valid and interesting provided there is good reasoning behind their use. 
We provide the `determine_model` method as an additional tool in this process, but do not claim that it should be the primary decision maker as to which model is most appropriate in a given situation.

## Stochastic Behavior
The functions `fit_hmm` and `determine_model` are stochastic in nature and also fairly slow, making them unsuitable for standard unit testing.
To understand the functionality of each method and ensure they are working as intended, a number of random tests were performed and their results analyzed.
These analyses are viewable at
[fit hmm distribution analysis](https://nbviewer.org/github/NTIA/gilbert-elliot-model/blob/main/tests/distribution_analysis.ipynb)
and
[determine model analysis](https://nbviewer.org/github/NTIA/gilbert-elliot-model/blob/main/tests/determine_model_tests.ipynb)
respectively.

## Command Line Interface
Upon installation, a command line tool called `gilbert-elliot.exe` is installed and added to the path.
It allows users to determine valid model control values and simulate error patterns.
To see all options for the command line tool run 

```
gilbert-elliot --help
```

### Examples

#### Valid region in three-parameter model
Here we fix the proportion of time spent in the Bad state, $\pi_B$, and the expected burst length, $L_1$, and then determine the valid region for the error rate, $\bar{x}$.
```
gilbert-elliot --pi_B 0.7 --L1 3 --free-parameter xbar
```
This outputs: `xbar must be in Interval.Lopen(0.466666666666667, 0.700000000000000)`.

#### Valid region in the four-parameter model
Here we fix the error rate, $\bar{x}$, the proportion of time spent in the Bad state, $\pi_B$, and the expected burst length, $L_1$, and then determine the valid region for the conditional error rate in the Bad state, $h$.
```
gilbert-elliot --xbar 0.6 --pi_B 0.7 --L1 3 --free-parameter h
```
This outputs: `h must be in Union(Interval.Ropen(0.142857142857143, 0.269069265858405), Interval.open(0.530930734141595, 0.571428571428572))`.

#### Translating error statistics into model controls
Building on the last example we select a value of $h$ from the valid region above and translate these error statistic model controls into the standard set of model parameters, $(p, r, k, h)$. 
```
gilbert-elliot --xbar 0.6 --pi_B 0.7 --L1 3 --h 0.55
```
This outputs: `Model Parameters: {'p': 0.166666666666667, 'r': 0.0714285714285714, 'h': 0.550000000000000, 'k': 0.0500000000000000}`.

#### Simulating an error pattern
This further builds on the previous example by using the model controls to generate a length 10000 error pattern and save it to `myerrors.csv`.
```
gilbert-elliot --xbar 0.6 --pi_B 0.7 --L1 3 --h 0.55 --simulate --n_observations 10000 --output myerrors.csv
```



## Contact
For questions contact Jaden Pieper, jpieper@ntia.gov
