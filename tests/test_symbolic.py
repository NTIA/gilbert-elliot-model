# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:28:26 2022

@author: jpieper
"""

import unittest
import numpy as np
import sympy as sp

import gilbert_elliot_model as ge

class TestSymbolic(unittest.TestCase):
    
    #--------------------
    # ----check_condition
    #--------------------
    
    def test_check_condition1g(self):
        x, y = sp.symbols('x y')
        expr_general = x + y
        expr = expr_general.subs({'y': 3})
        bound = 2
        operator = '>'
        param = x
        interval = sp.Interval(-np.inf, np.inf)
        
        expected = sp.Interval(-1, np.inf)
        result = ge.symbolic_solutions.check_condition(expr, bound, operator, param, interval)
        
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    def test_check_condition1ge(self):
        x, y = sp.symbols('x y')
        expr_general = x + y
        expr = expr_general.subs({'y': 3})
        bound = 2
        operator = '>='
        param = x
        interval = sp.Interval(-np.inf, np.inf)
        
        expected = sp.Interval(-1, np.inf)
        result = ge.symbolic_solutions.check_condition(expr, bound, operator, param, interval)
        
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    def test_check_condition1l(self):
        x, y = sp.symbols('x y')
        expr_general = x + y
        expr = expr_general.subs({'y': 3})
        bound = 2
        operator = 'l'
        param = x
        interval = sp.Interval(-np.inf, np.inf)
        
        expected = sp.Interval(-np.inf, -1)
        result = ge.symbolic_solutions.check_condition(expr, bound, operator, param, interval)
        
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    def test_check_condition1le(self):
        x, y = sp.symbols('x y')
        expr_general = x + y
        expr = expr_general.subs({'y': 3})
        bound = 2
        operator = '<='
        param = x
        interval = sp.Interval(-np.inf, np.inf)
        
        expected = sp.Interval(-np.inf, -1)
        result = ge.symbolic_solutions.check_condition(expr, bound, operator, param, interval)
        
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    #-----------------
    # ----solve_region
    #-----------------
    def test_solve_region1(self):
        L1 = sp.symbols('L1')
        ps = 0.8
        rs = 1/L1
        
        expected = sp.Interval(1, np.inf)
        result = ge.symbolic_solutions.solve_region(ps, rs, 'L1')
        
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    # Could do more but solve_region is tied so closely to the functionality of 
    # eval_param it feels reasonable to just focus on eval_param. solve_regions
    # is not used outside of eval_param and isn't really intended to...
    
    #---------------
    # ----eval_param
    #---------------
    
    def test_eval_param_xbarL1(self):
        # Two-parameter (xbar, L1) pair
        params = {
            'xbar': 0.6,
            'L1': 10,
            }
        expected = {'p': 0.15,
                    'r': 0.1,
                    'k': 1,
                    'h': 0,
                    }
        xbar, L1 = sp.symbols('xbar L1')
        p = xbar/(L1*(1 - xbar))
        r = 1/L1
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
    
    def test_eval_param_xbar_L1free(self):
        # Two-parameter (xbar, L1) pair
        params = {
            'xbar': 0.6,
            'L1': None,
            }
        expected = sp.Interval(1.5, np.inf, left_open=True, right_open=True)
        
        xbar, L1 = sp.symbols('xbar L1')
        p = xbar/(L1*(1 - xbar))
        r = 1/L1
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        param = 'L1'
        with self.subTest(parameter=param):
            self.assertEqual(expected, result)
    
    def test_eval_param_L1_xbarfree(self):
        # Two-parameter (xbar, L1) pair
        params = {
            'xbar': None,
            'L1': 4,
            }
        expected = sp.Interval(0, 0.8, left_open=True, right_open=True)
        xbar, L1 = sp.symbols('xbar L1')
        p = xbar/(L1*(1 - xbar))
        r = 1/L1
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        param = 'xbar'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
                
    def test_eval_param_xbargamma(self):
        # Two-parameter (xbar, gamma) pair
        params = {
            'xbar': 0.6,
            'gamma': 0.7,
            }
        expected = {'p': 0.18,
                    'r': 0.12,
                    'k': 1,
                    'h': 0,
                    }
        xbar, gamma = sp.symbols('xbar gamma')
        p = xbar*(1 - gamma)
        r = 1 - gamma - xbar*(1 - gamma)
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
                
    def test_eval_param_xbar_gammafree(self):
        # Two-parameter (xbar, gamma) pair
        params = {
            'xbar': 0.6,
            'gamma': None,
            }
        expected = sp.Interval(-2/3, 1, left_open=True, right_open=True)
    
        xbar, gamma = sp.symbols('xbar gamma')
        p = xbar*(1 - gamma)
        r = 1 - gamma - xbar*(1 - gamma)
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        param = 'gamma'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    def test_eval_param_gamma_xbarfree(self):
        # Two-parameter (xbar, gamma) pair
        params = {
            'xbar': None,
            'gamma': -0.2,
            }
        expected = sp.Interval(1/6, 5/6, left_open=True, right_open=True)
    
        xbar, gamma = sp.symbols('xbar gamma')
        p = xbar*(1 - gamma)
        r = 1 - gamma - xbar*(1 - gamma)
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        param = 'xbar'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
                
    def test_eval_param_xbarLR(self):
        # Two-parameter (xbar, LR) pair
        params = {
            'xbar': 0.6,
            'LR': 10,
            }
        expected = {'p': 0.06,
                    'r': 0.04,
                    'k': 1,
                    'h': 0,
                    }
        xbar, LR = sp.symbols('xbar LR')
        p = xbar/LR
        r = (1 - xbar)/LR
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
    
    def test_eval_param_xbar_LRfree(self):
        # Two-parameter (xbar, LR) pair
        params = {
            'xbar': 0.6,
            'LR': None,
            }
        expected = sp.Interval(6/10, np.inf, left_open=True, right_open=True)
        xbar, LR = sp.symbols('xbar LR')
        p = xbar/LR
        r = (1 - xbar)/LR
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        param = 'LR'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    def test_eval_param_LR_xbarfree(self):
        # Two-parameter (xbar, LR) pair
        params = {
            'xbar': None,
            'LR': 0.7,
            }
        expected = sp.Interval(3/10, 7/10, left_open=True, right_open=True)
        xbar, LR = sp.symbols('xbar LR')
        p = xbar/LR
        r = (1 - xbar)/LR
        result = ge.symbolic_solutions.eval_param(params=params, p=p, r=r)
        
        param = 'xbar'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    # Note: could write out full tests for all three- and four- pararm testsbut these are so 
    # similar to what tests for valid_regions would look like that it seems 
    # better to just focus on valid_regions tests
    
    #-------------------
    # ---- valid_regions
    #-------------------
    def test_valid_region_xbarL1(self):
        # Two-parameter (xbar, L1) pair
        params = {
            'xbar': 0.6,
            'L1': 10,
            }
        expected = {'p': 0.15,
                    'r': 0.1,
                    'k': 1,
                    'h': 0,
                    }
        
        result = ge.valid_regions(params=params)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
    
    def test_valid_regions_xbar_L1free(self):
        # Two-parameter (xbar, L1) pair
        params = {
            'xbar': 0.6,
            'L1': None,
            }
        expected = sp.Interval(1.5, np.inf, left_open=True, right_open=True)
        
        result = ge.valid_regions(params=params)
        
        param = 'L1'
        with self.subTest(parameter=param):
            self.assertEqual(expected, result)
    
    def test_valid_regions_L1_xbarfree(self):
        # Two-parameter (xbar, L1) pair
        params = {
            'xbar': None,
            'L1': 4,
            }
        expected = sp.Interval(0, 0.8, left_open=True, right_open=True)
        
        result = ge.valid_regions(params=params)
        
        param = 'xbar'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
                
    def test_valid_regions_xbargamma(self):
        # Two-parameter (xbar, gamma) pair
        params = {
            'xbar': 0.6,
            'gamma': 0.7,
            }
        expected = {'p': 0.18,
                    'r': 0.12,
                    'k': 1,
                    'h': 0,
                    }
        
        result = ge.valid_regions(params=params)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
                
    def test_valid_regions_xbar_gammafree(self):
        # Two-parameter (xbar, gamma) pair
        params = {
            'xbar': 0.6,
            'gamma': None,
            }
        expected = sp.Interval(-2/3, 1, left_open=True, right_open=True)
    
        result = ge.valid_regions(params=params)
        
        param = 'gamma'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    def test_valid_regions_gamma_xbarfree(self):
        # Two-parameter (xbar, gamma) pair
        params = {
            'xbar': None,
            'gamma': -0.2,
            }
        expected = sp.Interval(1/6, 5/6, left_open=True, right_open=True)
    
        result = ge.valid_regions(params=params)
        
        param = 'xbar'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
                
    def test_valid_regions_xbarLR(self):
        # Two-parameter (xbar, LR) pair
        params = {
            'xbar': 0.6,
            'LR': 10,
            }
        expected = {'p': 0.06,
                    'r': 0.04,
                    'k': 1,
                    'h': 0,
                    }
        
        result = ge.valid_regions(params=params)
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
    
    def test_valid_regions_xbar_LRfree(self):
        # Two-parameter (xbar, LR) pair
        params = {
            'xbar': 0.6,
            'LR': None,
            }
        expected = sp.Interval(6/10, np.inf, left_open=True, right_open=True)
        
        
        result = ge.valid_regions(params=params)
        
        param = 'LR'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    def test_valid_regions_LR_xbarfree(self):
        # Two-parameter (xbar, LR) pair
        params = {
            'xbar': None,
            'LR': 0.7,
            }
        expected = sp.Interval(3/10, 7/10, left_open=True, right_open=True)
        
        result = ge.valid_regions(params=params)
        
        param = 'xbar'
        
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    #---------------------------------
    #----valid_regions three-parameter
    #---------------------------------
    def test_valid_regions_xbarL1piB(self):
        params = {
            'xbar': 0.6,
            'L1': 3,
            'pi_B': 0.8,
            }
        expected = {
            'p': 4/9,
            'r': 1/9,
            'h': 1/4,
            'k': 1,
            }
        result = ge.valid_regions(params)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
                
    def test_valid_regions_xbarL1_piBfree(self):
        param = 'pi_B'
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'pi_B': None,
                  }
        
        expected = sp.Interval(6/10, 9/10)
        result = ge.valid_regions(params)
        
        param = 'pi_B' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    def test_valid_regions_xbarpi_B_L1free(self):
        
        
        params = {'xbar': 0.6,
                  'L1': None,
                  'pi_B': 0.8,
                  }
        
        expected = sp.Interval(16/7, 4)
        result = ge.valid_regions(params)
        
        param = 'L1' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    def test_valid_regions_L1pi_B_xbarfree(self):
        
        
        params = {'xbar': None,
                  'L1': 3,
                  'pi_B': 0.8,
                  }
        
        expected = sp.Interval(8/15, 32/45)
        result = ge.valid_regions(params)
        
        param = 'xbar' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    def test_valid_regions_xbarL1h(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'h': 0.2,
                  }
        
        expected = {'p': 1/2,
                    'r': 1/6
                    }
        result = ge.valid_regions(params)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)
    
    def test_valid_regions_xbarh_L1free(self):
        
        
        params = {'xbar': 0.6,
                  'L1': None,
                  'h': 0.2,
                  }
        
        expected = sp.Interval(15/7, 5)
        result = ge.valid_regions(params)
        
        param = 'L1' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    
    def test_valid_regions_L1h_xbarfree(self):
        
        
        params = {'xbar': None,
                  'L1': 3,
                  'h': 0.2,
                  }
        
        expected = sp.Interval(0, 24/35)
        result = ge.valid_regions(params)
        
        param = 'xbar' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    def test_valid_regions_xbarL1_hfree(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'h': None,
                  }
        
        expected = sp.Interval(0, 1/3)
        result = ge.valid_regions(params)
        
        param = 'h' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    #------------------------
    # ----(xbar, L1, pi_B, h)                
    #------------------------
    def test_valid_regions_xbarL1pi_Bh(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'pi_B': 0.7,
                  'h': 0.2,
                  }
        
        expected = {'p': 0.400000000000000,
                    'r': 0.171428571428571,
                    'h': 0.200000000000000,
                    'k': 0.866666666666667,
                    }
        result = ge.valid_regions(params)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)

    def test_valid_regions_xbarL1h_pi_Bfree(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'h': 0.2,
                  'pi_B': None
                  }
        
        expected = sp.Interval(0.5, 0.75)
        result = ge.valid_regions(params)
        
        param = 'pi_B' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )                
    
    
    
    def test_valid_regions_xbarL1pi_B_hfree(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'h': None,
                  'pi_B': 0.7
                  }
        
        expected = sp.Union(
            sp.Interval(0.142857142857143, 0.269069265858405, right_open=True), 
            sp.Interval(0.530930734141595, 0.571428571428572, left_open=True, right_open=True)
            )
        result = ge.valid_regions(params)
        
        param = 'h' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )  #---------------------------------
    
    def test_valid_regions_xbarpi_Bh_L1free(self):
        
        
        params = {'xbar': 0.6,
                  'L1': None,
                  'h': 0.2,
                  'pi_B': 0.7
                  }
        
        expected = sp.Interval(2.14285714285714, 4.09090909090909)
        result = ge.valid_regions(params)
        
        param = 'L1' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    def test_valid_regions_L1pi_Bh_xbarfree(self):
        params = {'xbar': None,
                  'L1': 3,
                  'h': 0.2,
                  'pi_B': 0.7
                  }
        
        expected = sp.Interval(0.560000000000000, 0.685714285714286)
        result = ge.valid_regions(params)
        
        param = 'xbar' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                        msg=(f'{result} not equivalent to expected: {expected}')
                                        )
    #------------------------
    # ----(xbar, L1, pi_B, k)
    #------------------------
    def test_valid_regions_xbarL1pi_Bk(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'pi_B': 0.7,
                  'k': 0.8,
                  }
        
        expected = {'p': 0.291666666666666,
                    'r': 0.125000000000000,
                    'h': 0.228571428571428,
                    'k': 0.800000000000000
                    }
        result = ge.valid_regions(params)
        
        for param, value in expected.items():
            with self.subTest(parameter=param):
                self.assertAlmostEqual(value, result[param], places=14)

    def test_valid_regions_xbarL1k_pi_Bfree(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'k': 0.8,
                  'pi_B': None
                  }
        
        expected = sp.Interval(0.5, 0.8)
        result = ge.valid_regions(params)
        
        param = 'pi_B' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )                
    
    
    
    def test_valid_regions_xbarL1pi_B_kfree(self):
        
        
        params = {'xbar': 0.6,
                  'L1': 3,
                  'k': None,
                  'pi_B': 0.7
                  }
        
        expected = sp.Union(
            sp.Interval(0, 0.0944949536696107, left_open=True, right_open=True), 
            sp.Interval(0.705505046330389, 1, left_open=True)
            )
        result = ge.valid_regions(params)
        
        param = 'k' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )  #---------------------------------
    
    def test_valid_regions_xbarpi_Bk_L1free(self):
        
        
        params = {'xbar': 0.6,
                  'L1': None,
                  'k': 0.8,
                  'pi_B': 0.7
                  }
        
        expected = sp.Interval(2.22727272727273, 3.50000000000000)
        result = ge.valid_regions(params)
        
        param = 'L1' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                       msg=(f'{result} not equivalent to expected: {expected}')
                                       )
    
    def test_valid_regions_L1pi_Bk_xbarfree(self):
        params = {'xbar': None,
                  'L1': 3,
                  'k': 0.8,
                  'pi_B': 0.7
                  }
        
        expected = sp.Interval(0.565444421759447, 0.738847795253392)
        result = ge.valid_regions(params)
        
        param = 'xbar' 
        with self.subTest(param="Number boundaries"):
            self.assertTrue(len(expected.boundary.args) == len(result.boundary.args),
                            msg=
                            (f'{result} not equivalent to expected: {expected}')
                            )
        expected_bounds = sorted(set(expected.boundary.args))
        result_bounds = sorted(set(result.boundary.args))
        
        for k, (result_bound, expected_bound) in enumerate(zip(result_bounds, expected_bounds)):
            with self.subTest(param=param, boundary=k):
                self.assertAlmostEqual(expected_bound, result_bound,
                                        msg=(f'{result} not equivalent to expected: {expected}')
                                        )
    
if __name__ == "__main__":
    unittest.main()