# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:14:57 2022

@author: jpieper
"""
import os
import time

import numpy as np
import pandas as pd

import gilbert_elliot_model as ge

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
p_fixed = 0.8
r_fixed = 0.6
k_fixed = 0.875
h_fixed = 0.234

check_trials = 50

def fit_hmm_two_param_distributions_fixed_values():
    print('--Testing two fixed parameter fit')
    p = p_fixed
    r = r_fixed
    k = 1
    h = 0
    init_params = {'k': k, 'h': h}
    repetitions = 1000
    estimations_list = []
    
    start_time = time.time()
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f'---{rep}/{repetitions}. Elapsed: {elapsed}')
            start_time = time.time()
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h)
        estimations = ge.fit_hmm(errors, init_params=init_params)
        estimations_list.append(estimations[0:4])
    estimations_df = pd.DataFrame(estimations_list, columns=['p', 'r', 'k', 'h'])
    fname = 'two_param_distributions_fixed_values.csv'
    fpath = os.path.join(__location__, fname)
    estimations_df.to_csv(fpath,
                         index=False,
                         )
def fit_hmm_two_param_distributions_random():
    print('--Testing two random parameter fit')
    k = 1
    h = 0
    
    rng = np.random.default_rng()
    init_params = {'k': k, 'h': h}
    repetitions = 1000
    estimations_list = []
    
    start_time = time.time()
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f'---{rep}/{repetitions}. Elapsed: {elapsed}')
            start_time = time.time()
        p, r = rng.uniform(low=0.01, high=0.99, size=2)
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h)
        estimations = ge.fit_hmm(errors, init_params=init_params)
        truth = np.array([p, r, k, h])
        difference = truth - np.array(estimations[0:4])
        
        estimations_list.append(difference)
    estimations_df = pd.DataFrame(estimations_list, 
                                  columns=['p_error', 'r_error', 'k_error', 'h_error'],
                                  )
    fname = 'two_param_distributions_random_errors.csv'
    fpath = os.path.join(__location__, fname)
    estimations_df.to_csv(fpath,
                         index=False,
                         )
def fit_hmm_three_param_distributions_fixed_values():
    print('--Testing three fixed parameter fit')
    p = p_fixed
    r = r_fixed
    k = 1
    h = h_fixed
    init_params = {'k': k}
    repetitions = 1000
    estimations_list = []
    
    start_time = time.time()
    
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f'---{rep}/{repetitions}. Elapsed: {elapsed}')
            start_time = time.time()
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h)
        estimations = ge.fit_hmm(errors, init_params=init_params)
        estimations_list.append(estimations[0:4])
    estimations_df = pd.DataFrame(estimations_list, columns=['p', 'r', 'k', 'h'])
    fname = 'three_param_distributions_fixed_values.csv'
    fpath = os.path.join(__location__, fname)
    estimations_df.to_csv(fpath,
                         index=False,
                         )
def fit_hmm_three_param_distributions_random():
    print('--Testing three random parameter fit')
    k = 1
    
    rng = np.random.default_rng()
    init_params = {'k': k}
    repetitions = 1000
    estimations_list = []
    
    start_time = time.time()
    times = []
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            remaining_est = np.mean(times)/check_trials*(repetitions-rep)
            print(f'---{rep}/{repetitions}. Estimated remaining time: {remaining_est} s')
            start_time = time.time()
        p, r, h = rng.uniform(low=0.01, high=0.99, size=3)
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h)
        estimations = ge.fit_hmm(errors, init_params=init_params)
        truth = np.array([p, r, k, h])
        difference = truth - np.array(estimations[0:4])
        out_vars = []
        out_vars.extend(truth)
        out_vars.extend(estimations[0:4])
        out_vars.extend(difference)
        
        estimations_list.append(out_vars)
    estimations_df = pd.DataFrame(estimations_list, 
                                  columns=[
                                      'p_target',
                                      'r_target',
                                      'k_target',
                                      'h_target',
                                      'p_estimate',
                                      'r_estimate',
                                      'k_estimate',
                                      'h_estimate',
                                      'p_error',
                                      'r_error',
                                      'k_error',
                                      'h_error'
                                      ],
                                  )
    fname = 'three_param_distributions_random_errors.csv'
    fpath = os.path.join(__location__, fname)
    estimations_df.to_csv(fpath,
                         index=False,
                         )
    
def fit_hmm_four_param_distributions_fixed_values():
    print('--Testing four fixed parameter fit')
    p = p_fixed
    r = r_fixed
    k = k_fixed
    h = h_fixed
    
    repetitions = 1000
    estimations_list = []
    
    start_time = time.time()
    times = []
    
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            remaining_est = np.mean(times)/check_trials*(repetitions-rep)
            print(f'---{rep}/{repetitions}. Estimated remaining time: {remaining_est} s')
            start_time = time.time()
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h)
        estimations = ge.fit_hmm(errors)
        estimations_list.append(estimations[0:4])
    estimations_df = pd.DataFrame(estimations_list, columns=['p', 'r', 'k', 'h'])
    fname = 'four_param_distributions_fixed_values.csv'
    fpath = os.path.join(__location__, fname)
    estimations_df.to_csv(fpath,
                         index=False,
                         )
def fit_hmm_four_param_distributions_random():
    
    print('--Testing four random parameter fit')
    rng = np.random.default_rng()
    repetitions = 1000
    estimations_list = []
    
    start_time = time.time()
    times = []
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            remaining_est = np.mean(times)/check_trials*(repetitions-rep)
            print(f'---{rep}/{repetitions}. Estimated remaining time: {remaining_est} s')
            start_time = time.time()
        p, r, k, h = rng.uniform(low=0.01, high=0.99, size=4)
        if k < h:
            k, h= (h, k)
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h)
        estimations = ge.fit_hmm(errors)
        truth = np.array([p, r, k, h])
        difference = truth - np.array(estimations[0:4])
        
        difference = truth - np.array(estimations[0:4])
        out_vars = []
        out_vars.extend(truth)
        out_vars.extend(estimations[0:4])
        out_vars.extend(difference)
        estimations_list.append(out_vars)
    estimations_df = pd.DataFrame(estimations_list, 
                                  columns=[
                                      'p_target',
                                      'r_target',
                                      'k_target',
                                      'h_target',
                                      'p_estimate',
                                      'r_estimate',
                                      'k_estimate',
                                      'h_estimate',
                                      'p_error',
                                      'r_error',
                                      'k_error',
                                      'h_error'
                                      ],
                                  )
    fname = 'four_param_distributions_random_errors.csv'
    fpath = os.path.join(__location__, fname)
    estimations_df.to_csv(fpath,
                         index=False,
                         )
    


def determine_model_two_param():
    rng = np.random.default_rng()
    repetitions = 300
    
    start_time = time.time()
    times = []
    k = 1
    h = 0
    trials = []
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0 and rep != 0:
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            remaining_est = np.mean(times)/check_trials*(repetitions - rep)
            print(f'---{rep}/{repetitions}. Elapsed: {np.round(elapsed)} s. Estimated remaining time: {np.round(remaining_est)} s')
            start_time = time.time()
        p, r = rng.uniform(low=0.01, high=0.99, size=2)
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h, n=1000)
        best_model, model_dict = ge.determine_model(errors)
        trial = {
            'model_type': 'two_param',
            'p_target': p,
            'r_target': r,
            'k_target': k,
            'h_target': h,
            'model_est': best_model,
            }
        params = ['p', 'r', 'k', 'h']
        for model_name, model in model_dict.items():
            model_key = '_' + model_name
            for ix, param in enumerate(params):
                trial[param + model_key] = model['model'][ix]
            trial['score' + model_key] = model['score']
        trials.append(trial)
    trial_df = pd.DataFrame(trials)
    fname = 'determine_model_two_param.csv'
    fpath = os.path.join(__location__, fname)
    trial_df.to_csv(fpath,
                 index=False,
                 )
    
def determine_model_three_param():
    rng = np.random.default_rng()
    repetitions = 300
    
    start_time = time.time()
    times = []
    k = 1
    trials = []
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0 and rep != 0:
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            remaining_est = np.mean(times)/check_trials*(repetitions - rep)
            print(f'---{rep}/{repetitions}. Elapsed: {np.round(elapsed)} s. Estimated remaining time: {np.round(remaining_est)} s')
            start_time = time.time()
        p, r, h = rng.uniform(low=0.01, high=0.99, size=3)
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h, n=1000)
        best_model, model_dict = ge.determine_model(errors)
        trial = {
            'model_type': 'three_param',
            'p_target': p,
            'r_target': r,
            'k_target': k,
            'h_target': h,
            'model_est': best_model,
            }
        params = ['p', 'r', 'k', 'h']
        for model_name, model in model_dict.items():
            model_key = '_' + model_name
            for ix, param in enumerate(params):
                trial[param + model_key] = model['model'][ix]
            trial['score' + model_key] = model['score']
        trials.append(trial)
    trial_df = pd.DataFrame(trials)
    fname = 'determine_model_three_param.csv'
    fpath = os.path.join(__location__, fname)
    trial_df.to_csv(fpath,
                 index=False,
                 )
def determine_model_four_param():
    rng = np.random.default_rng()
    repetitions = 300
    
    start_time = time.time()
    times = []
    
    trials = []
    for rep in range(repetitions):
        if np.mod(rep, check_trials) == 0 and rep != 0:
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            remaining_est = np.mean(times)/check_trials*(repetitions - rep)
            print(f'---{rep}/{repetitions}. Elapsed: {np.round(elapsed)} s. Estimated remaining time: {np.round(remaining_est)} s')
            start_time = time.time()
        p, r, k, h = rng.uniform(low=0.01, high=0.99, size=4)
        if k < h:
            k, h= (h, k)
        errors = ge.gilbert_elliot.simulate_errors(p=p, r=r, k=k, h=h, n=1000)
        best_model, model_dict = ge.determine_model(errors)
        trial = {
            'model_type': 'four_param',
            'p_target': p,
            'r_target': r,
            'k_target': k,
            'h_target': h,
            'model_est': best_model,
            }
        params = ['p', 'r', 'k', 'h']
        for model_name, model in model_dict.items():
            model_key = '_' + model_name
            for ix, param in enumerate(params):
                trial[param + model_key] = model['model'][ix]
            trial['score' + model_key] = model['score']
        trials.append(trial)
    trial_df = pd.DataFrame(trials)
    fname = 'determine_model_four_param.csv'
    fpath = os.path.join(__location__, fname)
    trial_df.to_csv(fpath,
                 index=False,
                 )

if __name__ == "__main__":
    readme_fname = 'Fixed_Parameter_README.md'
    readme_fpath = os.path.join(__location__, readme_fname)
    fid = open(readme_fpath, 'w')
    msg = ('Fixed parameter values are as follows. Note that if a test is '
           'a two-parameter fit k=1 and h=0 are fixed, for three-parameter k=1 '
           'is fixed. Otherwise parameters are set to:\n'
           'p,r,k,h\n'
           f'{p_fixed},{r_fixed},{k_fixed},{h_fixed}'
           )
    fid.write(msg)
    fid.close()
    
    two_param_fixed_fname = 'two_param_distributions_fixed_values.csv'
    two_param_random_fname = 'two_param_distributions_random_errors.csv'
    if not os.path.exists(os.path.join(__location__, two_param_fixed_fname)):
        fit_hmm_two_param_distributions_fixed_values()
    
    if not os.path.exists(os.path.join(__location__, two_param_random_fname)):
        fit_hmm_two_param_distributions_random()
        
    three_param_fixed_fname = 'three_param_distributions_fixed_values.csv'
    three_param_random_fname = 'three_param_distributions_random_errors.csv'
    if not os.path.exists(os.path.join(__location__, three_param_fixed_fname)):
        fit_hmm_three_param_distributions_fixed_values()
    
    if not os.path.exists(os.path.join(__location__, three_param_random_fname)):
        fit_hmm_three_param_distributions_random()
        
    four_param_fixed_fname = 'four_param_distributions_fixed_values.csv'
    four_param_random_fname = 'four_param_distributions_random_errors.csv'
    if not os.path.exists(os.path.join(__location__, four_param_fixed_fname)):
        fit_hmm_four_param_distributions_fixed_values()
    
    if not os.path.exists(os.path.join(__location__, four_param_random_fname)):
        fit_hmm_four_param_distributions_random()
    
    two_param_determine_fname = 'determine_model_two_param.csv'
    if not os.path.exists(os.path.join(__location__, two_param_determine_fname)):
        determine_model_two_param()
    
    three_param_determine_fname = 'determine_model_three_param.csv'
    if not os.path.exists(os.path.join(__location__, three_param_determine_fname)):
        determine_model_three_param()
        
    four_param_determine_fname = 'determine_model_four_param.csv'
    if not os.path.exists(os.path.join(__location__, four_param_determine_fname)):
        determine_model_four_param()
    