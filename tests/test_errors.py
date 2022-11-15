# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:36:04 2022

@author: jpieper
"""
import unittest
import numpy as np


import gilbert_elliot_model as ge

class TestErrorSimulations(unittest.TestCase):
    '''
    Tests for functions in gilbert_elliot.py.
    
    Note that many of these tests work on functions that are random and can only
    be evaluated with general statistical tests that work MOST of the time but 
    not always. 95% confidence intervals are used to test statistics, meaning 
    any test that relies on them should fail around 5% of the time.
    '''
    
    # The hmm fit process is a challenging problem requiring lots of data to get
    # reasonable answers and they still may be a bit off from the truth. 
    # Balance time for tests by decreasing hmm_fits_n and increasing acceptable
    # tolerance with hmm_delta
    # Note that even if you set hmm_fit_n to a very high number these tests may
    # still fail as both the three-parameter and four-parameter models are 
    # complciated processes with many different sets of parameters capable of 
    # creating similar looking data
    hmm_delta = 0.1
    hmm_fit_n = 1000
    
    sim_reps = 100
    # --------------------------------
    # ---- model_error_statistics ----
    # --------------------------------
    def test_model_error_stats1(self):
        # Test two-param independent errors case
        p = 0.2; r = 0.8; k = 1; h = 0
        
        error_rate = 0.2
        expected_burst_length = 1.25
        lag_one_correlation = 0
        relative_expected_burst_length = 1
        expected_errors = {
            'error_rate': error_rate,
            'bad_proportion': error_rate,
            'expected_burst_length': expected_burst_length,
            'lag_one_correlation': lag_one_correlation,
            'relative_expected_burst_length': relative_expected_burst_length,
            }
        
        error_stats = ge.model_error_statistics(p, r, k, h)
        for key, value in expected_errors.items():
            with self.subTest(stat=key):
                self.assertAlmostEqual(value, error_stats[key])
    
    def test_model_error_stats2(self):    
        # Test two-param errors case
        p = 0.3; r = 0.5; k = 1; h = 0
        
        error_rate = 0.375
        expected_burst_length = 2
        lag_one_correlation = 0.2
        relative_expected_burst_length = 1.25
        expected_errors = {
            'error_rate': error_rate,
            'bad_proportion': error_rate,
            'expected_burst_length': expected_burst_length,
            'lag_one_correlation': lag_one_correlation,
            'relative_expected_burst_length': relative_expected_burst_length,
            }
        
        error_stats = ge.model_error_statistics(p, r, k, h)
        for key, value in expected_errors.items():
            with self.subTest(stat=key):
                self.assertAlmostEqual(value, error_stats[key])
    
    def test_model_error_stats3(self):
        # Test four-parameter independent errors case
        p = 0.2; r = 0.8; k = 0.8; h = 0.3
        
        # bad_proportion = p/(p + r)
        bad_proportion = 0.2
        # error_rate = (1 - k)*(1 - bad_proportion) + (1 - h)*bad_proportion
        error_rate = 0.3
        
        # expected_burst_length = ((1 - k)*r + (1 - h)*p)/((1 - k)*r*((1 - p)*k + p*h) + (1 - h)*p*((1 - r)*h + r*k))
        expected_burst_length = 1.4285714285714282
        
        expected_errors = {
            'error_rate': error_rate,
            'bad_proportion': bad_proportion,
            'expected_burst_length': expected_burst_length,
            }
        
        error_stats = ge.model_error_statistics(p, r, k, h)
        for key, value in expected_errors.items():
            with self.subTest(stat=key):
                self.assertAlmostEqual(value, error_stats[key])
    
    # -------------------------------
    # ---- data_error_statistics ----
    # -------------------------------
    def test_data_error_stats1(self):
        # alternating errors
        n = 50
        errors = int(n/2)*[1, 0]
        error_stats = ge.gilbert_elliot.data_error_statistics(errors)
        
        stat = 'error_rate'
        with self.subTest(stat=stat):
            self.assertEqual(0.5, error_stats[stat])
            
        stat = 'expected_burst_length'
        with self.subTest(stat=stat):
            self.assertEqual(1, error_stats[stat])
            
    def test_data_error_stats2(self):
        # All bursts are length 3
        n = 100
        errors = int(n/5)*[1, 1, 1, 0, 0]
        error_stats = ge.gilbert_elliot.data_error_statistics(errors)
        
        stat = 'error_rate'
        with self.subTest(stat=stat):
            self.assertEqual(0.6, error_stats[stat])
            
        stat = 'expected_burst_length'
        with self.subTest(stat=stat):
            self.assertEqual(3, error_stats[stat])
            
    def test_data_error_stats3(self):
        # Simple bursts to count
        errors = []
        max_burst_len = 10
        zero_choices = np.arange(1, max_burst_len + 1)
        rng = np.random.default_rng()
        
        for burst_len in np.arange(1, max_burst_len + 1):
            zero_len = rng.choice(zero_choices)
            errors.extend(zero_len*[0])
            errors.extend(burst_len * [1])
        
        error_stats = ge.gilbert_elliot.data_error_statistics(errors)
        
        stat = 'error_rate'
        with self.subTest(stat=stat):
            self.assertEqual(np.mean(errors), error_stats[stat])
            
        stat = 'expected_burst_length'
        expected_burst_length = ((max_burst_len + 1)*max_burst_len/2)/max_burst_len
        with self.subTest(stat=stat):
            self.assertEqual(expected_burst_length, error_stats[stat])
    def test_data_error_stats4(self):
        errors = [
            1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 
            1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 
            1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
            0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0
            ]
        error_stats = ge.gilbert_elliot.data_error_statistics(errors)
        
        stat = 'error_rate'
        with self.subTest(stat=stat):
            self.assertEqual(0.46, error_stats[stat])
            
        stat = 'expected_burst_length'
        with self.subTest(stat=stat):
            self.assertEqual(1.5862068965517242, error_stats[stat])
    #----------------------------
    # ---- conditional_event ----
    # ---------------------------
    
    def test_conditional_event1(self):
        
        event = ge.gilbert_elliot.conditional_event(1, np.random.default_rng())
        self.assertEqual(1, event)
    
    def test_conditional_event2(self):
        
        event = ge.gilbert_elliot.conditional_event(0, np.random.default_rng())
        self.assertEqual(0, event)
        
    def test_conditional_event3(self):
        n = 10000
        events = []
        rng = np.random.default_rng()
        event_rate = 0.7
        for k in range(n):
            event = ge.gilbert_elliot.conditional_event(event_rate, rng)
            events.append(event)
        measured_event_rate = np.mean(events)
        measured_event_CI = measured_event_rate + np.array([-1, 1])*1.96*np.std(events)/np.sqrt(len(events))
        self.assertTrue(measured_event_CI[0] < event_rate and event_rate < measured_event_CI[1],
                        msg=('95% confidence interval does not contain expected value'
                        f'. {event_rate} not in {measured_event_CI}.')
                        )
    
    # -------------------------
    # ---- simulate_errors ----
    # -------------------------
    def test_simulate_errors_alternate(self):
        n = 50
        expected_errors = int(n/2)*[1, 0]
        # A seed is necessary as there's a 50/50 prob of starting in G or B here
        errors = ge.gilbert_elliot.simulate_errors(p=1, r=1, k=1, h=0, n=n, seed=12345)
        
        self.assertListEqual(expected_errors, errors)
        
    
    def test_simulate_errors_0s(self):
        errors = ge.gilbert_elliot.simulate_errors(0.8, 0.6, 1, 1)
        
        expected_errors = len(errors)*[0]
        self.assertListEqual(expected_errors, errors)
    
    def test_simulate_errors_1s(self):
        errors = ge.gilbert_elliot.simulate_errors(0.8, 0.6, 0, 0)
        
        expected_errors = len(errors)*[1]
        self.assertListEqual(expected_errors, errors)
    
    def test_simulate_errors_B(self):
        errors = ge.gilbert_elliot.simulate_errors(1, 0, 1, 0)
        expected_errors = len(errors)*[1]
        self.assertListEqual(expected_errors, errors)
        
    
    def test_simulate_errors_G(self):
        errors = ge.gilbert_elliot.simulate_errors(0, 1, 1, 0)
        expected_errors = len(errors)*[0]
        self.assertListEqual(expected_errors, errors)
        
    def test_simulate_errors_fixed_seed(self):
        expected_errors = [
            1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 
            1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 
            1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
            0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0
            ]
        errors = ge.gilbert_elliot.simulate_errors(0.8, 0.6, 0.95, 0.3, n=100, seed=12345)
        
        self.assertListEqual(errors, expected_errors)
    
    # ------------------
    # ---- simulate ----
    # ------------------
    
    def test_simulate_1(self):
        
        params = {'p': 0.8, 'r': 0.6, 'k': 0.95, 'h': 0.3}
        error_rates = []
        burst_lengths = []
        error_stats= ge.model_error_statistics(p=params['p'],
                                               r=params['r'], 
                                               k=params['k'], 
                                               h=params['h'],
                                               )
        for r in range(self.sim_reps):    
            errors = ge.simulate(params=params, n_obs=1000)
            meas_stats = ge.data_error_statistics(errors)
            error_rates.append(meas_stats['error_rate'])
            burst_lengths.append(meas_stats['expected_burst_length'])
        
        ci_quantile = [0.025, 0.975]
        meas_stats = {
            'error_rate_95%_confidence_interval': np.quantile(error_rates, ci_quantile),
            'expected_burst_length_95%_confidence_interval': np.quantile(burst_lengths, ci_quantile)}    
        
        stats = ['error_rate', 'expected_burst_length']
        for stat in stats:
            stat_ci_name = stat + '_95%_confidence_interval'
            stat_ci = meas_stats[stat_ci_name]
            with self.subTest(statistic=stat):
                self.assertTrue(
                    stat_ci[0] < error_stats[stat] and error_stats[stat] < stat_ci[1],
                    msg=('95% confidence interval does not contain expected value. '
                         f'{error_stats[stat]} not in {stat_ci}.'
                         )
                    )
    def test_simulate_2(self):
        params = {'xbar': 0.4, 'L1': 7}
        
        error_rates = []
        burst_lengths = []
        for r in range(self.sim_reps):    
            errors = ge.simulate(params=params, n_obs=1000)
            meas_stats = ge.data_error_statistics(errors)
            error_rates.append(meas_stats['error_rate'])
            burst_lengths.append(meas_stats['expected_burst_length'])
        
        ci_quantile = [0.025, 0.975]
        meas_stats = {
            'error_rate_95%_confidence_interval': np.quantile(error_rates, ci_quantile),
            'expected_burst_length_95%_confidence_interval': np.quantile(burst_lengths, ci_quantile)}    
        
        
        param2stat = {'xbar': 'error_rate',
                      'L1': 'expected_burst_length',
                      }
        for param, stat in param2stat.items():
            stat_ci_name = stat + '_95%_confidence_interval'
            stat_ci = meas_stats[stat_ci_name]
            with self.subTest(statistic=stat):
                self.assertTrue(
                    stat_ci[0] < params[param] and params[param] < stat_ci[1],
                    msg=('95% confidence interval does not contain expected value. '
                         f'{params[param]} not in {stat_ci}.'
                         )
                    )
         
    def test_simulate_3(self):
        params = {'xbar': 0.4, 'L1': 3, 'pi_B': 0.45}
            
        error_rates = []
        burst_lengths = []
        for r in range(self.sim_reps):    
            errors = ge.simulate(params=params, n_obs=1000)
            meas_stats = ge.data_error_statistics(errors)
            error_rates.append(meas_stats['error_rate'])
            burst_lengths.append(meas_stats['expected_burst_length'])
        
        ci_quantile = [0.025, 0.975]
        meas_stats = {
            'error_rate_95%_confidence_interval': np.quantile(error_rates, ci_quantile),
            'expected_burst_length_95%_confidence_interval': np.quantile(burst_lengths, ci_quantile)}    
        
        
        param2stat = {'xbar': 'error_rate',
                      'L1': 'expected_burst_length',
                      }
        for param, stat in param2stat.items():
            stat_ci_name = stat + '_95%_confidence_interval'
            stat_ci = meas_stats[stat_ci_name]
            with self.subTest(statistic=stat):
                self.assertTrue(
                    stat_ci[0] < params[param] and params[param] < stat_ci[1],
                    msg=('95% confidence interval does not contain expected value. '
                         f'{params[param]} not in {stat_ci}.'
                         )
                    )
    def test_simulate_4(self):
        params = {'xbar': 0.4, 'L1': 3, 'pi_B': 0.45, 'h': 0.12}
            
        error_rates = []
        burst_lengths = []
        for r in range(self.sim_reps):    
            errors = ge.simulate(params=params, n_obs=1000)
            meas_stats = ge.data_error_statistics(errors)
            error_rates.append(meas_stats['error_rate'])
            burst_lengths.append(meas_stats['expected_burst_length'])
        
        ci_quantile = [0.025, 0.975]
        meas_stats = {
            'error_rate_95%_confidence_interval': np.quantile(error_rates, ci_quantile),
            'expected_burst_length_95%_confidence_interval': np.quantile(burst_lengths, ci_quantile)}    
        
        
        param2stat = {'xbar': 'error_rate',
                      'L1': 'expected_burst_length',
                      }
        for param, stat in param2stat.items():
            stat_ci_name = stat + '_95%_confidence_interval'
            stat_ci = meas_stats[stat_ci_name]
            with self.subTest(statistic=stat):
                self.assertTrue(
                    stat_ci[0] < params[param] and params[param] < stat_ci[1],
                    msg=('95% confidence interval does not contain expected value. '
                         f'{params[param]} not in {stat_ci}.'
                         )
                    )
    # -----------------
    # ---- fit_hmm ----
    # -----------------
    # Note most of the fit_hmm testing is covered by distribution_tests.py and 
    # the accompanying analysis in distribution_analysis.{ipynb, html}
    def test_fit_hmm1(self):
        '''fit two-parameter model'''
        p = 0.8
        r = 0.6
        k = 1
        h = 0
        errors = ge.gilbert_elliot.simulate_errors(p, r, k, h, self.hmm_fit_n)
        
        init_params= {'k': k, 'h': h}
        p_est, r_est, k_est, h_est, model_est = ge.gilbert_elliot.fit_hmm(errors, init_params=init_params)
        
        params = {'p': [p, p_est],
                  'r': [r, r_est],
                  'k': [k, k_est],
                  'h': [h, h_est],
                  }
        for param, values in params.items():    
            with self.subTest(parameter=param):
                self.assertAlmostEqual(values[0], values[1], delta=self.hmm_delta)
                
    # def test_fit_hmm2(self):
    #     '''fit three-parameter model'''
    #     p = 0.8
    #     r = 0.6
    #     k = 1
    #     h = 0.3
    #     errors = ge.gilbert_elliot.simulate_errors(p, r, k, h, self.hmm_fit_n)
        
    #     init_params= {'k': k}
    #     p_est, r_est, k_est, h_est, model_est = ge.gilbert_elliot.fit_hmm(errors, init_params=init_params)
        
    #     params = {'p': [p, p_est],
    #               'r': [r, r_est],
    #               'k': [k, k_est],
    #               'h': [h, h_est],
    #               }
    #     for param, values in params.items():    
    #         with self.subTest(parameter=param):
    #             self.assertAlmostEqual(values[0], values[1], delta=self.hmm_delta)
    
    
    # def test_fit_hmm3(self):
    #     '''fit four-parameter model'''
    #     p = 0.8
    #     r = 0.6
    #     k = 1
    #     h = 0
    #     errors = ge.gilbert_elliot.simulate_errors(p, r, k, h, self.hmm_fit_n)
        
    #     init_params= {'k': k, 'h': h}
    #     p_est, r_est, k_est, h_est, model_est = ge.gilbert_elliot.fit_hmm(errors, init_params=init_params)
        
    #     params = {'p': [p, p_est],
    #               'r': [r, r_est],
    #               'k': [k, k_est],
    #               'h': [h, h_est],
    #               }
    #     for param, values in params.items():    
    #         with self.subTest(parameter=param):
    #             self.assertAlmostEqual(values[0], values[1], delta=self.hmm_delta)
    
    # def test_fit_hmm4(self):
    #     '''fit two-parameter model deterministic alternating 1s and 0s'''
    #     p = 1
    #     r = 1
    #     k = 1
    #     h = 0
    #     errors = ge.gilbert_elliot.simulate_errors(p, r, k, h, self.hmm_fit_n)
        
    #     init_params= {'k': k, 'h': h}
    #     p_est, r_est, k_est, h_est, model_est = ge.gilbert_elliot.fit_hmm(errors, init_params=init_params)
        
    #     params = {'p': [p, p_est],
    #               'r': [r, r_est],
    #               'k': [k, k_est],
    #               'h': [h, h_est],
    #               }
    #     for param, values in params.items():    
    #         with self.subTest(parameter=param):
    #             self.assertEqual(values[0], values[1])
    # # -------------------------
    # # ---- determine_model ----
    # # -------------------------
    # def test_determine_model_two_param(self):
    #     p = 0.8
    #     r = 0.6
    #     k = 1
    #     h = 0
    #     errors = ge.gilbert_elliot.simulate_errors(p, r, k, h, self.hmm_fit_n)
    #     argmax, models, scores = ge.determine_model(errors)
    #     self.assertEqual(argmax, 2, 
    #                      msg=f'Model incorrectly identified. 2 - two-param, 1 - three-param, 0 - four-param'
    #                      )
    
    #--------------------------
    # ----determine_model tests
    #--------------------------
    
    # Note that determine_model tests are handeld by distribution_tests.py and 
    # the associated analysis in determine_model_tests.{ipynb, html}
    
    
    
    
if __name__ == '__main__':
    unittest.main()