# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:13:13 2022

@author: jpieper
"""

import argparse
import copy

import numpy as np

from hmmlearn import hmm

from .symbolic_solutions import valid_regions, valid_parameters
#------------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------------
NotFullyDefinedParameterSet = ValueError(
    ('Unable to simulate as a free parameter was given. A fully defined'
     ' set of parameters must be used as input.'
     )
     )
#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def model_error_statistics(p, r, k, h):
    """
    Determine error statistics given model parameters.

    Determines the proportion of time spent in the Bad state, the error rate, 
    and the expected burst length of error patterns generated with the given 
    Gilbert-Elliot burst error model parameters. In the case of the two-parameter
    model, where k=1 and h=0, the lag one correlation and relative expected
    burst error length are also included in the error_stats dict.
    
    Parameters
    ----------
    p : float
        Probability of transitioning from the Good state to the Bad state.
    r : float
        Probability of transitioning from the Bad state to the Good state.
    k : float
        Probability of no error occurring when in the Good state.
    h : float
        Probability of no error occurring when in the Bad state.

    Returns
    -------
    error_stats : dict
        Dictionary for relevant error statistics
    """

    bad_proportion = p/(p + r)
    error_rate = ((1 - k) * r + (1 - h) * p)/(p + r)
    expected_burst_length = ((1 - k) * r + (1 - h) * p)\
        /((1 - k) * r * ((1 - p) * k + p*h) + (1 - h) * p * ((1 - r) * h + r*k))
    error_stats = {'bad_proportion': bad_proportion,
                   'error_rate': error_rate,
                   'expected_burst_length': expected_burst_length,
                   }
    if k == 1 and h == 0:
        # In two-parameter model we will calculate lag-one correlation and
        # relative expected burst length
        lag_one_correlation = 1 - r - p
        relative_expected_burst_length = (1 - error_rate)/r

        error_stats['lag_one_correlation'] = lag_one_correlation
        error_stats['relative_expected_burst_length'] = relative_expected_burst_length

    return error_stats

def data_error_statistics(error):
    """
    Measure error statsitics from error pattern.

    Measures the error_rate and expected_burst_length of the error pattern.

    Parameters
    ----------
    error : list
        List of 0s and 1s representing an error pattern.

    Returns
    -------
    error_stats : dict
        Dictionary containing error statistic estimates. A list containing
        all observed burst lengths is also included.

    """
    error_rate = np.mean(error)

    # Uniquely count observed burst lengths
    burst_lengths = []
    burst_count = 0
    for e in error:
        if e == 1:
            # We have an error, increment burst count
            burst_count += 1
        elif burst_count > 0:
            # We were in a burst but it is over now, save its length
            burst_lengths.append(burst_count)
            burst_count = 0

    if burst_count > 0:
        burst_lengths.append(burst_count)

    average_burst_length = np.mean(burst_lengths)

    error_stats = {
        'error_rate': error_rate,
        'expected_burst_length': average_burst_length,
        'burst_lengths': burst_lengths,
        }
    return error_stats

def conditional_event(conditional_event_rate, random_number_generator):
    """
    Evaluate stochastic event based off of the conditional_event_rate.

    An event is represented by a 1 and no event a 0. For example if the
    conditional_event_rate = 0.7, that means that 70% of the time this function
    will return a 1 and 30% of the time a 0.

    Parameters
    ----------
    conditional_event_rate : float
        Rate at which the event occurs and a 1 is generated.
    random_number_generator : numpy.random._generator.Generator
        Random number generator used to generate uniform random variables.

    Returns
    -------
    error : int
        Either 0 or 1, 1 represents the event occuring.

    """
    event_draw = random_number_generator.uniform()

    if event_draw < conditional_event_rate:
        event = 1
    else:
        event = 0

    return event

def simulate_errors(p, r, k, h, n=1000, seed=None):
    '''
    Simulate errors using the Gilbert-Elliot burst error model.

    Creates a list representing an error signal. The error signal is expected
    to be added with modulo 2 addition to a binary signal, so 0 means no error
    occured in that element and a 1 means an error did occur.

    Parameters
    ----------
    p : float
        Probability of transitioning from the Good state to the Bad state.
    r : float
        Probability of transitioning from the Bad state to the Good state.
    k : float
        Probability of no error occurring when in the Good state.
    h : float
        Probability of no error occurring when in the Bad state.
    n : int, optional
        Length of the output array. The default is 1000.

    Returns
    -------
    errors : list
        List of errors generated by the model, where 0 represents no error,
        and 1 represents an error.

    '''
    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Deterimine initial state based off of proportion of time spent in bad state
    pi_B = p/(p + r)
    init_state = conditional_event(pi_B, rng)

    # states will be a list of 0s and 1s where 0 represents the Good state and
    # B the bad state. Thus the conditional_event_probability of transitioning
    # from G to B will be p, so we will use conditional_event(p, rng) for
    # transition evaluations in G.
    # Conversely the conditional event probability of transitioning from B to B
    # will be 1 - r so we will use conditional_event(1 - r, rng) to handle
    # transition evaluations in the Bad state.
    states = [init_state]

    # Determine if an error occured in the initial state
    errors = []

    # Iterate to generate entire error vector
    for i in np.arange(1, n):
        # Evaluate previous state to determine approprite conditional probabilities
        if states[i - 1] == 0:
            # We're in state G
            # Do a draw to see if we have an error
            errors.append(conditional_event(1 - k, rng))

            # Determine our next state
            new_state = conditional_event(p, rng)

        else:
            # We're in state B
            # Do a draw to see if we have an error
            errors.append(conditional_event(1 - h, rng))

            # Determine our next state
            new_state = conditional_event(1 - r, rng)

        states.append(new_state)
    # For last state evaluate error
    if states[n-1]== 0:
        errors.append(conditional_event(1 - k, rng))

    else:
        errors.append(conditional_event(1 - h, rng))

    return errors


def fit_hmm(obs, n_fits=100, n_iter=100, init_params=None):
    """
    Fit two-state Hidden Markov model to data in observation.

    Fits a two-state Hidden Markov model to a series of observations. To avoid
    local maxima the the fit is performed a number of times and the most likely
    model is returned by computing the "score" of each output, which is the
    log probability of observation under the model, i.e., return the model most
    likely to emit the observations.

    Parameters
    ----------
    obs : np.ndarray
        Array of observations to fit model to. If provided as a list it will be
        converted to an np.ndarray with reshape(1, -1).
    n_fits : int, optional
        Number of unique fits performed. Increasing this parameter will
        increase the time this function takes to return but should increase
        accuracy up to a point by avoiding local maxima. The default is 100.
    n_iter : int, optional
        Maximum number of iterations performed in the hmmlearn fit process.
        The default is 100.
    init_params : dict, optional
        Dictionary of initial parameters. Can contain pi to initialize the
        starting condition, p and r (must both be included) to initalize the
        probability transition matrix of the HMM, k and h to initialize both
        conditional probabilities of error to fit a two-parameter model, or just k
        to fit a three-parameter model. The default is None.

    Returns
    -------
    best_p : float
        Estimate for probability of transitioning from Good state to Bad.
    best_r : float
        Estimate for probability of transitioning from Bad state to Good.
    best_k : float
        Estimate for probability of no error occuring in the Good state.
    best_h : float
        Estimate for probability of no error occuring in the Bad state.
    best_model : hmmlearn.hmm.CategoricalHMM
        Full output model given obs and init_params.

    """

    # Initialize model
    # MultinomialHMM correct in hmmlearn < 0.2.7 but 0.2.8 updated class name and broke backwards compatibility
    # fit_model = hmm.MultinomialHMM(n_components=2, n_iter=n_iter) 
    fit_model = hmm.CategoricalHMM(n_components=2, n_iter=n_iter, implementation='scaling')
    # Two-state model (good and bad states)
    fit_model.n_features = 2

    # Prepare any pre-initialized parameters
    used_params = fit_model.init_params
    if init_params is not None:
        if 'pi' in init_params:
            init_pi = init_params['pi']
            used_params = used_params.replace('s', '')
        if 'p' in init_params and 'r' in init_params:
            init_A = np.array([[1 - init_params['p'], init_params['p']],
                              [init_params['r'], 1 - init_params['r']]
                              ])
            used_params = used_params.replace('t', '')
        if 'k' in init_params:
            if 'h' in init_params:
                ih = init_params['h']
            else:
                # Start h at 0.5
                rng = np.random.default_rng()
                # Start with random h guess
                ih = rng.uniform()
            init_B = np.array([[init_params['k'], 1 - init_params['k']],
                               [ih, 1 - ih]
                               ])
            used_params = used_params.replace('e', '')

    # init_params can be any combination of 's' 't' 'e' in a single string for
    # startprob, transmat, and other characters for sublcass-specific emission
    # parameters. See hmmlearn.hmm.BaseHMM for more details.
    fit_model.init_params = used_params

    # Place to store best model we've seen
    best_model = None
    best_score = -np.inf

    for guess_ix in range(n_fits):
        # Prep for a fresh fit
        if 's' in fit_model.init_params:
            if hasattr(fit_model, 'startprob_'):
                # Delete so we don't get annoying print about overwriting
                del fit_model.startprob_
        else:
            # Set to initial guess
            fit_model.startprob_ = init_pi

        if 't' in fit_model.init_params:
            if hasattr(fit_model, 'transmat_'):
                del fit_model.transmat_
        else:
            fit_model.transmat_ = init_A

        if 'e' in fit_model.init_params:
            if hasattr(fit_model, 'emissionprob_'):
                del fit_model.emissionprob_
        else:
            fit_model.emissionprob_ = init_B

        # Attempt a fit
        try:
            fit_model.fit(obs)
        except ValueError:
            # Try reshaping the observations to expected type if we saw a ValueError
            obs = np.array(obs).reshape(1, -1)
            fit_model.fit(obs)

        # Update best model and score if necessary
        if best_score < fit_model.score(obs):
            best_score = fit_model.score(obs)
            best_model = copy.deepcopy(fit_model)
    # Pull out the things we care about
    best_p = best_model.transmat_[0, 1]
    best_r = best_model.transmat_[1, 0]
    best_k = best_model.emissionprob_[0, 0]
    best_h = best_model.emissionprob_[1, 0]
    if best_k < best_h:
        # Enforce k is greater so that "Good" state preserves meaning
        best_k, best_h = best_h, best_k
        best_p, best_r = best_r, best_p

    return best_p, best_r, best_k, best_h, best_model

def determine_model(obs, n_fits=100, n_iter=100):
    '''
    Determine most likely version of Gilbert-Elliot model given observation.

    Fits three versions of the Gilbert-Elliot model. The two-parameter case where
    p and r are estimated and k=1 and h=0 are fixed, the three-parameter case
    where p, r, and h are estimated and k=1 is fixed, and the four-parameter case
    where p, r, k, and h are all estimated. This is accomplished by passing values
    of k and h in as initial parameters when appropriate to the model. The best
    version is determined by comparing the "score" of each output, which is the
    log probability of the observation under the model.

    It is important to note that there may be more than one set of  model
    parameters yield a given error pattern or very similar error patterns. As
    such, this method is not intended to be the primary decision driver for
    users looking to decide which model to use. It is best to only use a more
    complicated model when there is good reason to suspect it is appropriate.
    This method is included and intended only as an additional decision-making
    tool.

    Parameters
    ----------
    obs : np.array
        Array of observations.
    n_fits : int, optional
        Number of fits performed for each model determination. The fitting
        algorithm can only determine local maxima, so a number of fits must be
        estimated from a variety of initial conditions to try to find the best
        one that is ideally closer to the global maxima. The default is 100.
    n_iter : int, optional
        Maximum number of iterations per fit. The default is 100.

    Returns
    -------
    best_model : str
        Dictionary key of most likely model.
    outs : dict
        Dictionary containing keys for each model type: two_param, three_param,
        and four_param. Each value for a given key is a dict with a 'model' key
        representing the output from gilbert_elliot_model.fit_hmm and a 'score'
        key with the score of the given model for the given observation.
    '''
    if isinstance(obs, list):
        obs = np.array(obs).reshape(1, -1)
    # Fit the full four-parameter model
    four_param = fit_hmm(obs, n_fits=n_fits, n_iter=n_iter)
    # Fix the initial guess for k to be 1 to get a three-parameter fit
    three_param = fit_hmm(obs, init_params={'k': 1}, n_fits=n_fits, n_iter=n_iter)
    # Fix the initial guess for k=1 and h=0 to get a two-parameter fit
    two_param = fit_hmm(obs, init_params={'k': 1, 'h': 0}, n_fits=n_fits, n_iter=n_iter)

    # Initialize dict with models and no scores
    outs = {'four_param': {'model': four_param, 'score': None},
            'three_param': {'model': three_param, 'score': None},
            'two_param': {'model': two_param, 'score': None},
            }
    # Variable to track best model
    best_fit = {'model_name': None, 'score': -np.inf}
    for model_name, model in outs.items():
        # Get score from hmm fit
        model['score'] = model['model'][-1].score(obs)
        # Update our best if necessary
        if model['score'] > best_fit['score']:
            best_fit['model_name'] = model_name
            best_fit['score'] = model['score']


    best_model = best_fit['model_name']

    return best_model, outs


def simulate(params, n_obs=1000):
    '''
    Simulate errors using the Gilbert-Elliot burst model and a variety of controls as parameters.

    Creates a list representing an error signal. The error signal is expected
    to be added with modulu 2 addition to a binary signal, so 0 means no error
    and 1 means an error occured in that element. Translates error statistics
    such as average loss rate or expected burst length to standard
    Gilbert-Elliot model parameters (p, r, k, h) prior to simulations.

    Any set of parameters in `gilbert_elliot_model.valid_parameters(pairs=True)`
    can be used.

    Parameters
    ----------
    params : dict
        Dictionary with keys of parameter names and values. Valid parameter
        pairs can be seen with
        `gilbert_elliot_model.valid_parameters(pairs=True)`.

    Returns
    -------
    errors : list
        List of errors generated by the model, where 0 represents no error,
        and 1 represents an error.

    '''

    model_params = valid_regions(params)
    if not isinstance(model_params, dict):
        raise NotFullyDefinedParameterSet

    for param_name, param in model_params.items():
        if param < 0 or param > 1:
            raise ValueError(
                ('Invalid parameters selected. Caused model parameters that are'
                 f' not probabilities: {model_params}.'
                 )
                )
    if model_params['h'] is None:
        h = 0
    else:
        h = model_params['h']
    if model_params['k'] is None:
        k = 1
    else:
        k = model_params['k']
    p = model_params['p']
    r = model_params['r']
    # Simulate
    errors = simulate_errors(p=p, r=r, k=k, h=h, n=n_obs)
    return errors


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments for model controls
    parser.add_argument(
        '--p',
        type=float,
        help='Probability of transitioning from Good state to Bad state. Must be in (0, 1)'
        )
    parser.add_argument(
        '--r',
        type=float,
        help='Probability of transitioning from Bad state to Good state. Must be in (0, 1)'
        )
    parser.add_argument(
        '--h',
        type=float,
        help='Conditional success rate in the bad/burst state. Must be in [0, 1).'
        )
    parser.add_argument(
        '-k',
        type=float,
        help='Conditional success rate in the good state. Must be in (0, 1].'
        )
    parser.add_argument(
        '--xbar', '-x',
        type=float,
        help='Average loss rate for model. Must be in (0, 1).'
        )
    parser.add_argument(
        '--L1', '-L',
        type=float,
        help='Average length of error burst. Must be greater than 1, i.e. in (1, infinity).'
        )
    parser.add_argument(
        '--pi_B',
        type=float,
        help='Proportion of time spent in the bad/burst state. Must be in (0, 1).'
        )
    parser.add_argument(
        '--gamma', '-g',
        type=float,
        help='Normalized loss covariance. Must be in (-1, 1).'
        )

    parser.add_argument(
        '-f', '--free-parameter',
        type=str,
        help='Parameter to keep free. Will solve for a valid range for this based off other parameters passed.'
        )

    parser.add_argument(
        '-s', '--simulate',
        action='store_true',
        help='Simulate error stream given current parameters and save it to --output'
        )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='gilbert-elliot-simulation.csv',
        help='File name for saving simulated error streams'
        )
    parser.add_argument(
        '-n', '--n_observations',
        type=int,
        default=1000,
        help='Number of observations in output simulation.'
        )

    # Parse arguments
    args = parser.parse_args()

    # Initialize set to store parameters
    params = {}
    # Get list of valid parameters
    valid_parameters_list = valid_parameters(pairs=False)

    for k, v in args.__dict__.items():
        # Check if parameter given and is valid
        if v is not None and k in valid_parameters_list:
            params[k] = v

    # Store the free parameter if we have
    if args.free_parameter is not None:
        params[args.free_parameter] = None

    # Determine valid regions
    results = valid_regions(params)

    if isinstance(results, dict):
        print(f'Model Parameters: {results}')
    else:
        print(f'{args.free_parameter} must be in {results}')
    if args.simulate and isinstance(results, dict):
        # Run simulations
        errors = simulate(results, n_obs=args.n_observations)
        # Save results
        np.savetxt(args.output, errors, delimiter=',', fmt='%d')
    elif args.simulate and args.free_parameter is not None:
        raise NotFullyDefinedParameterSet

# -----------------------------------------------------------------------------
# ---- __main__
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()