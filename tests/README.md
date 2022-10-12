# gilbert-elliot-model tests
## Unit tests
Most of the functions have tests written using `unittest`. These tests are structed as follows:
* `test_errors.py` - tests for functions in `gilbert_elliot.py`.
* `test_symbolic.py` - tests for functions in `sybmolic_solutions.py`.

Some of these tests cover stochastic methods that are fast enough to run many replications. This allows us to generate 95% confidence intervals with ease. It is important to note that for these tests we expect them to fail roughly 5% of the time they are run. They are intended to be used with expert judgement as to whether the expected failure rate is being exceeded.

## Distribution Analysis
Some of the functions are both stochastic and slow, making them challenging to run replications for on the fly, and are thus unsuitable for unit tests. Instead using `distribution_tests.py` we have run many replications for these methods and saved the results in csv files corresponding to each test.

In particular the methods tested are `gilbert_elliot_model.fit_hmm` and `gilbert_elliot_model.determine_model`. Written analysis of the functionality/performance of these methods are included in two separate jupyter notebook, `distribution_analysis.ipynb` and `determine_model_tests.ipynb`. These include discussions of the methodology for these tests.

The two jupyter notebooks are intended to be viewed via at:
* [fit hmm distribution analysis](https://nbviewer.org/github/NTIA/gilbert-elliot-model/blob/main/tests/distribution_analysis.ipynb)
* [determine model analysis](https://nbviewer.org/github/NTIA/gilbert-elliot-model/blob/main/tests/determine_model_tests.ipynb)