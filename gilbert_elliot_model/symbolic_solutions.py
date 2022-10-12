# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:18:45 2022

@author: jpieper
"""

import sympy as sp
import numpy as np

import warnings

# Valid ranges for model parameters and error statistics
intervals = {
    'xbar': sp.Interval(0, 1),
    'L1': sp.Interval(1, np.inf),
    'LR': sp.Interval(0.5, np.inf),
    'gamma': sp.Interval(-1, 1),
    'pi_B': sp.Interval(0, 1),
    'p': sp.Interval(0, 1),
    'r': sp.Interval(0, 1),
    'h': sp.Interval(0, 1),
    'k': sp.Interval(0, 1),
    }

def eval_param(params, p, r, h=None, k=None):
    '''
    Substitute values into symbolic expressions and evaluate them.

    Intended to substitute error statistic controls into expressions for
    controls for the Gilbert-Elliot burst error model. Takes a dictionary of
    controls and substitutes those values into expressions of model statistics,
    p, r, k, and h.

    Parameters
    ----------
    params : dict
        Dictionary of controls for Gilbert-Elliot model.
    p : sympy.core.mul.Mul
        Probability of transitioning from Good state to Bad, written in terms
        of parameters defined in params.
    r : sympy.core.mul.Mul
        Probability of transitioning from Bad state to Good, written in terms
        of parameters defined in params.
    h : sympy.core.mul.Mul, optional
        Probability of no error when in Bad state, written in terms of
        parameters defined in params. The default is None, used in
        two-parameter solutions.
    k : sympy.core.mul.Mul, optional
        Probability of no werror when in Good state, written in terms of
        parameters defined in params. The default is None, used in two- and
        three-parameter solutions.

    Raises
    ------
    ValueError
        Parameters must be in their valid regions.

    Returns
    -------
    restrict : dict or sympy.sets.sets.Interval
        DESCRIPTION.

    '''
    # Variables to track how many parameters were given and which one is free
    given_count = 0
    free_param = None

    # Initialize substituted forms of expressions
    subs_expr = {'p': p,
                 'r': r,
                 }

    if h is not None:
        subs_expr['h'] = h
    else:
        subs_expr['h'] = None
    if k is not None:
        # ks = k
        subs_expr['k'] = k
    else:
        subs_expr['k'] = None

    for param, val in params.items():
        # Increment given parameter count
        if val is not None:
            if val not in intervals[param]:
                raise ValueError(f'{param} must be in {intervals[param]}')
            given_count += 1
        else:
            free_param = param

        # Substitute param for value
        for sub_var, sub in subs_expr.items():
            if sub is not None:
                subs_expr[sub_var] = sub.subs({param: val})


    # Ensure at least all but one parameters fixed
    if given_count < len(params) - 1:
        raise ValueError('Must fix all or all but one parameter')
    elif given_count == len(params):
        # If all fixed display answer
        if subs_expr['h'] is None:
            subs_expr['h'] = 0
        if subs_expr['k'] is None:
            subs_expr['k'] = 1
        # print(f'Model Parameters: {subs_expr}')
        restrict = subs_expr
    else:
        # Set up for solving for final region
        restrict = solve_region(ps=subs_expr['p'],
                                rs=subs_expr['r'],
                                free_param=free_param,
                                hs=subs_expr['h'],
                                ks=subs_expr['k'])
        # print(f'{free_param} must be in: {restrict}')
    return restrict

def check_condition(expr, bound, operator, param, interval, check=''):
    '''
    Check that expression validates an operator and bound when solved for param.

    Using sympy.solveset, evaluates "expr operator bound", e.g., 'k < 1'.

    Parameters
    ----------
    expr : sympy.core.mul.Mul
        sympy expression to be evalauted against bound using operator.
    bound : float
        Condition that the expression will be checked against with operator.
    operator : str
        One of ['>', '>=', '<', '<=']. Used to compare expression to bound.
    param : sympy.core.symbol.Symbol
        Parameter to solve for when evaluating the full expression with bound.
    interval : sympy.sets.sets.Interval
        Valid region for param.
    check : str, optional
        Optional message displayed in the event that the solution results in an
        EmptySet and a ValueError is raised. The default is ''.

    Raises
    ------
    ValueError
        Raised when the EmptySet is generated as the only valid solution.

    Yields
    ------
    cond : sympy.sets.sets.Interval
        Interval in which param satisfies the expression, operator, bound
        combination.

    '''
    if operator in ['>', 'g', 'greater']:
        full_expr = expr > bound
    elif operator in ['>=', 'ge', 'greaterequal']:
        full_expr = expr >= bound
    elif operator in ['<', 'l', 'less']:
        full_expr = expr < bound
    elif operator in ['<=', 'le', 'lessequal']:
        full_expr = expr <= bound
    cond = sp.solveset(full_expr, param, interval)

    if cond == sp.EmptySet:
        raise ValueError(
            (f'Input parameters do not yield valid results when solving {check}. '
             f'Solving for {param}. '
             f'Comparing `{expr}` {operator} `{bound}`')
            )
    return cond

def solve_region(ps, rs, free_param, hs=None, ks=None,):
    '''
    Solve for the valid region for a free parameter.

    Expressions for each parameter of the Gilbert-Elliot model (p, r, k, h)
    written in terms of error statistic controls with at least all but one error
    statistic substitued for a value. It then solves for the valid region for
    the remaining parameter, given by free_param, by enforcing the basic
    constraints that p, r, k, and h are probabilities so
    0 < p, r, k, h < 1 must hold.

    Parameters
    ----------
    ps : sympy.core.mul.Mul
        Substituted expression for p, where values for input error statistics
        have been substituted into the expression.
    rs : sympy.core.mul.Mul
        Substituted expression for r, where values for input error statistics
        have been substituted into the expression.
    hs : sympy.core.mul.Mul
        Substituted expression for h, where values for input error statistics
        have been substituted into the expression.
    ks : sympy.core.mul.Mul
        Substituted expression for k, where values for input error statistics
        have been substituted into the expression.
    fp : str
        String representation of free parameter remaining in any of the
        substituted expressions for ps, rs, ks, and/or hs. Must be one of
        'xbar, L1, LR, gamma, pi_B, p, r, h, k'. This list is determined by the
        global variable intervals.

    Returns
    -------
    restrict : sympy.sets.sets.Interval
        Interval for free_param that satisfies the conditions that
        0 < p, r, k, h < 1.

    '''
    # Symbol for free parameter
    fp = sp.symbols(free_param)
    # Interval for free parameter
    interval = intervals[free_param]

    # We always have p and r, solve for their valid regions
    p0 = check_condition(expr=ps, operator='>', bound=0, param=fp,
                         interval=interval, check='0 < p')
    p1 = check_condition(expr=ps, operator='<', bound=1, param=fp,
                         interval=interval, check='p < 1')
    r0 = check_condition(expr=rs, operator='>', bound=0, param=fp,
                         interval=interval, check='0 < r')
    r1 = check_condition(expr=rs, operator='<', bound=1, param=fp,
                         interval=interval, check='r < 1')

    # Store regions in list
    regions = [p0, p1, r0, r1]

    if hs is not None:
        # Add h regions if we're using h
        # Note that h can be 0
        h0 = check_condition(expr=hs, operator='>=', bound=0, param=fp,
                             interval=interval, check='0 <= h')
        h1 = check_condition(expr=hs, operator='<', bound=1, param=fp,
                             interval=interval, check='h < 1')
        regions.append(h0)
        regions.append(h1)
    if ks is not None:
        # Add k regions if we're using k
        k0 = check_condition(expr=ks, operator='>', bound=0, param=fp, interval=interval, check='0 < k')
        # Note that k can be 1
        k1 = check_condition(expr=ks, operator='<=', bound=1, param=fp, interval=interval, check='k <= 1')
        regions.append(k0)
        regions.append(k1)

    # Initialize region to be -infinity to infinity (fully inclusive)
    restrict = sp.Interval(-np.inf, np.inf)
    for reg in regions:
        # Restrict by intersecting with each region we've found
        restrict = restrict.intersect(reg)
    if restrict == sp.EmptySet:
        warnings.warn(f'Input parameters do not yield any valid values for {free_param}')
    return restrict

def valid_parameters(pairs=True):
    '''
    Return valid parameters either as a list of pairs or a set of all parameters.

    Parameter pairs are one of the following: model parameter for the two-,
    three-, or four-parameter Gilbert-Elliot burst error model, e.g., (p, r),
    (p, r, h), or (p, r, k, h) respectively. Or they are pairs of error-statistics
    that can be translated into the standard model parameters and function as
    model controls.

    Returns
    -------
    list
        If pairs is True returns a list of all the valid sets of parameters. If
        pairs is False returns a list containing each valid parameter just once.

    '''
    #--------------------------------------------------------------------------
    # ---- Valid parameter pair list
    #--------------------------------------------------------------------------
    # These will be uncommented as they are implemented
    valid_parameter_pairs = [
        # ----- 2 parameter pairs
        {'p', 'r'},
        {'xbar', 'L1'},
        {'xbar', 'LR'},
        {'xbar', 'gamma'},
        # ----- 3 parameter pairs
        {'p', 'r', 'h'},
        {'xbar', 'L1', 'pi_B'},
        {'xbar', 'L1', 'h'},
        # ----- 4 parameter pairs
        {'p', 'r', 'k', 'h'},
        {'xbar', 'L1', 'pi_B', 'h'},
        {'xbar', 'L1', 'pi_B', 'k'},
        # {'xbar', 'L1', 'pi_G', 'k'},
        ]
    if pairs:
        return valid_parameter_pairs
    else:
        out_list = set()
        for pair in valid_parameter_pairs:
            for v in pair:
                out_list.add(v)
        return list(out_list)

def check_standard_model_parameters(input_params):
    for par, val in input_params.items():
        if val is None:
            raise ValueError('Standard model parameters cannot be set as free parameters')

def valid_regions(params):
    '''
    Determine valid ranges for Gilbert-Elliot model controls and error statistics.

    Translates valid parameter pairs, as defined by
    gilbert_elliot_model.valid_parameters(pairs=True), into standard model
    parameters for the Gilbert-Elliot burst error model. If one value in params
    is set to None, it is treated as a free-parameter and the valid regions for
    that parameter are solved for and returned. If a full set of controls are in
    params the standard model controls are returned.

    Note that only one error statistic control can be set as a free parameter.

    Parameters
    ----------
    params : dict
        Dictionary with keys of parameter names and values for set values.
        Use None to keep parameter free.


    Returns
    -------
    restrict : dict or sympy.Interval
        If a full set of model controls are passed a dict containing the keys
        'p', 'r', 'k', 'h' and associated values is returned as controls for the
        Gilbert-Elliot model. If one control is set as a free-parameter a sympy
        Interval is returned containing the valid regions for that free-parameter
        to ensure all model controls are valid probabilities.

    '''


    input_params = set(params)
    valid_parameter_pairs = valid_parameters(pairs=True)
    if input_params not in valid_parameter_pairs:
        raise ValueError(f'Invalid set of parameters. Must be one of {valid_parameter_pairs}, received \'{params}\'')

    # -------------------------------------------------------------------------
    # ---- Parameter Expressions
    # -------------------------------------------------------------------------
    # ----two-parameter pairs
    if input_params == {'p' , 'r'}:
        check_standard_model_parameters(params)

        restrict = params
        restrict['h'] = 0
        restrict['k'] = 1

    elif input_params == {'xbar', 'L1'}:
        xbar, L1 = sp.symbols('xbar L1')
        p = xbar/(L1*(1 - xbar))
        r =  1/L1

        restrict = eval_param(params, p, r)
    elif input_params == {'xbar', 'LR'}:

        xbar, LR = sp.symbols('xbar LR')
        p = xbar/LR
        r = (1 - xbar)/LR

        restrict = eval_param(params, p, r)

    elif input_params == {'xbar', 'gamma'}:
        
        xbar, gamma = sp.symbols('xbar gamma')
        p = xbar*(1 - gamma)
        r = 1 - gamma - xbar*(1 - gamma)

        restrict = eval_param(params, p, r)

    # ----three-parameter pairs
    elif input_params == {'p', 'r', 'h'}:
        check_standard_model_parameters(params)
        restrict = params
        restrict['k'] = 1


    elif input_params == {'xbar', 'L1', 'pi_B'}:

        xbar, L1, pi_B = sp.symbols('xbar L1 pi_B')

        h = 1 - xbar/pi_B
        r = (xbar * L1 - pi_B*(L1 - 1))/(xbar*L1)
        p = pi_B/(1 - pi_B) * r

        restrict = eval_param(params, p, r, h)
    elif input_params == {'xbar', 'L1', 'h'}:

        xbar, L1, h = sp.symbols('xbar L1 h')

        r = (1 - L1*h)/(L1*(1 - h))
        p = (xbar*(1 - L1*h))/(L1*(1 - h)*(1 - h - xbar))

        restrict = eval_param(params, p, r, h=h)

    # ----four-parameter pairs
    elif input_params == {'p', 'r', 'k', 'h'}:
        check_standard_model_parameters(params)
        restrict = params

    elif input_params == {'xbar', 'L1', 'pi_B', 'h'}:

        xbar, L1, pi_B, h = sp.symbols('xbar L1 pi_B h')

        p = (L1*(xbar - pi_B*(1 - h)) * (h*pi_B + xbar - 1) + (1 - pi_B)*(xbar - L1*h*pi_B*(1 - h)))/(L1*(h + xbar - 1)**2)
        r = (1 - pi_B)/pi_B * p
        k = (1 - xbar - h*pi_B)/(1 - pi_B)

        restrict = eval_param(params, p, r, h=h, k=k)
    elif input_params == {'xbar', 'L1', 'pi_B', 'k'}:

        xbar, L1, pi_B, k = sp.symbols('xbar L1 pi_B k')

        r = 1 + (pi_B*(xbar -L1*(k*(k + 2*(xbar - 1)) - xbar + 1)))/(L1 * (1 - xbar - k)**2)
        p = pi_B/(1 - pi_B) * r
        h = (1 - k*(1 - pi_B) - xbar)/pi_B

        restrict = eval_param(params, p, r, h=h, k=k)
    else:
        raise ValueError(f'Unknown set of parameters: {params}')

    return restrict
