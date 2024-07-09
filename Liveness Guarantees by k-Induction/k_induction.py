'''
Using k-Induction for verifying a liveness property for the 2D docking problem presented in the paper "Formally Verifying Deep Reinforcement Learning Controllers with Lyapunov Barrier Certificates", https://arxiv.org/abs/2405.14058 .

Goal: 
    To verify decreasing distance property.
    ###The new property to check is:
    #|x| + |y| decreases OR
    #|v_x| + |v_y| decreases

Method:
    We check if after all iterations, 
    the negation of the property holds, encoded as:
    "sum of all |x| + |y|  prev" is >= "sum of all |x| + |y|  cur"
    AND
    "sum of all |v_x| + |v_y|  prev" is >= "sum of all |v_x| + |v_y|  cur";
    where "prev" indicates the initial state and the "next" indicates the last state.

Expected result: UNSAT.
    We assume there is an N amount of steps to be taken in order to dock successfully.
    If we get UNSAT with the lowest number of N, the task is completed

Make sure to run "python3 -u" command.

@created 2024 02 28
@last modified 2024 07 09
@auth Ieva, Udayan, Andrew
'''

#!/usr/bin/env python3
import math
import numpy as np
import time

from maraboupy import Marabou
from maraboupy import MarabouCore



start = time.perf_counter()

NUMBER_OF_STEPS = 10

x_lb = 0.51
x_ub = 5.0
y_lb = 0.51
y_ub = 5.0

v_x_lb = -0.2
v_x_ub = 0.2
v_y_lb = -0.2
v_y_ub = 0.2


m = 12
n = 0.001027
t = 1


# Matrix encoding of system dynamics
coeffs_x_t = [
    4 - 3 * np.cos(n * t),
    0,
    1 / n * np.sin(n * t),
    2 / n - 2 / n * np.cos(n * t),
    (1 - np.cos(n * t)) / (m * n ** 2),
    2 * t / (m * n) - 2 * np.sin(n * t) / (m * n ** 2),
    -1
]
coeffs_y_t = [
    -6 * n * t + 6 * np.sin(n * t),
    1,
    -2 / n + 2 / n * np.cos(n * t),
    -3 * t + 4 / n * np.sin(n * t),
    (-2 * t) / (m * n) + (2 * np.sin(n * t)) / (m * n ** 2),
    4 / (m * n ** 2) - (3 * t ** 2) / (2 * m) - (4 * np.cos(n * t)) / (m * n ** 2),
    -1
]
coeffs_v_x_t = [
    3 * n * np.sin(n * t),
    0,
    np.cos(n * t),
    2 * np.sin(n * t),
    np.sin(n * t) / (m * n),
    2 / (m * n) - (2 * np.cos(n * t)) / (m * n),
    -1
]
coeffs_v_y_t = [
    -6 * n + 6 * n * np.cos(n * t),
    0,
    -2 * np.sin(n * t),
    -3 + 4 * np.cos(n * t),
    (2 * np.cos(n * t) - 2) / (m * n),
    (-3 * t) / (m) + (4 * np.sin(n * t)) / (m * n),
    -1
]

options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=5000, numWorkers=10, tighteningStrategy="sbt", solveWithMILP=True) 
networkName = "ckpt_200_l1_reward_with_pre_post_processing.onnx" 

# INITIALIZATION
epsilon = 1e-5 

clip_lb = -1
clip_ub = 1


for i in range(0,NUMBER_OF_STEPS):
    print('Trying to dock with Max Number of Steps N = ', NUMBER_OF_STEPS)

    network = Marabou.read_onnx(networkName)
    prev_state = network.inputVars[0][0]
    prev_control = network.outputVars[0][0]

    x_initial = prev_state[0]
    y_initial = prev_state[1]
    v_x_initial = prev_state[2]
    v_y_initial = prev_state[3]
    
    network.setLowerBound(x_initial, x_lb)
    network.setUpperBound(x_initial, x_ub)
    network.setLowerBound(y_initial, y_lb)
    network.setUpperBound(y_initial, y_ub)
    network.setLowerBound(v_x_initial, v_x_lb)
    network.setUpperBound(v_x_initial, v_x_ub)
    network.setLowerBound(v_y_initial, v_y_lb)
    network.setUpperBound(v_y_initial, v_y_ub)
    
    #Property verification begins here

    #|x| initial
    x_mod_initial = network.getNewVariable()
    network.addAbsConstraint(x_initial, x_mod_initial) 

    #|y| initial
    y_mod_initial = network.getNewVariable()
    network.addAbsConstraint(y_initial, y_mod_initial) 

    #|x|+|y| for initial state
    xy_mod_initial = network.getNewVariable()
    network.addEquality([x_mod_initial, y_mod_initial, xy_mod_initial], [1,1,-1], 0, False)


    #|v_x| initial
    v_x_mod_initial = network.getNewVariable()
    network.addAbsConstraint(v_x_initial, v_x_mod_initial) 

    #|v_y| initial
    v_y_mod_initial = network.getNewVariable()
    network.addAbsConstraint(v_y_initial, v_y_mod_initial) 

    #|v_x|+|v_y| for initial state
    v_xy_mod_initial = network.getNewVariable()
    network.addEquality([v_x_mod_initial, v_y_mod_initial, v_xy_mod_initial], [1,1,-1], 0, False)


    
    for j in range(i+1):
        network.shallowClear()
        network.readONNX(networkName, None, None, reindexOutputVars=False)
    
        cur_state = network.inputVars[j+1][0]
        x_cur, y_cur, v_x_cur, v_y_cur = cur_state[0], cur_state[1], cur_state[2], cur_state[3]
        x_prev, y_prev, v_x_prev, v_y_prev = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
        F_x_prev, F_y_prev = prev_control[0], prev_control[1]

        print("i=", i, ", j=", j)

        e3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e3.addAddend(1.0, x_prev)
        e3.setScalar(0.5)

        e4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e4.addAddend(1.0, y_prev)
        e4.setScalar(0.5)

        e5 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e5.addAddend(1.0, x_prev)
        e5.setScalar(-0.5)

        e6 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e6.addAddend(1.0, y_prev)
        e6.setScalar(-0.5)

        network.addDisjunctionConstraint([[e3],[e4],[e5],[e6]])

        #handling F_x_clipping 
        aux1_prev = network.getNewVariable()
        F_x_clip_prev = network.getNewVariable()
        network.addEquality([aux1_prev,F_x_prev],[1,-1],-1*clip_lb,False) 
        aux2_prev = network.getNewVariable()
        network.addRelu(aux1_prev,aux2_prev) 
        aux3_prev = network.getNewVariable()
        network.addEquality([aux3_prev,aux2_prev],[1,-1],clip_lb,False) 
        aux4_prev = network.getNewVariable()
        network.addEquality([aux4_prev,aux3_prev],[-1,-1],-clip_ub,False) 
        aux5_prev = network.getNewVariable()
        network.addRelu(aux4_prev,aux5_prev) 
        network.addEquality([F_x_clip_prev,aux5_prev],[-1,-1],-clip_ub,False) 

        #handling F_y_clipping
        aux1_prev_y = network.getNewVariable()
        F_y_clip_prev = network.getNewVariable()
        network.addEquality([aux1_prev_y,F_y_prev],[1,-1],-1*clip_lb,False) 
        aux2_prev_y = network.getNewVariable()
        network.addRelu(aux1_prev_y,aux2_prev_y) 
        aux3_prev_y = network.getNewVariable()
        network.addEquality([aux3_prev_y,aux2_prev_y],[1,-1],clip_lb,False) 
        aux4_prev_y = network.getNewVariable()
        network.addEquality([aux4_prev_y,aux3_prev_y],[-1,-1],-clip_ub,False) 
        aux5_prev_y = network.getNewVariable()
        network.addRelu(aux4_prev_y,aux5_prev_y) 
        network.addEquality([F_y_clip_prev,aux5_prev_y],[-1,-1],-clip_ub,False) 

        # System dynamics for final step
        vars_x_cur = [x_prev, y_prev, v_x_prev, v_y_prev, F_x_clip_prev, F_y_clip_prev, x_cur]
        network.addEquality(vars_x_cur, coeffs_x_t, 0, False)

        vars_y_cur = [x_prev, y_prev, v_x_prev, v_y_prev, F_x_clip_prev, F_y_clip_prev, y_cur]
        network.addEquality(vars_y_cur, coeffs_y_t, 0, False)

        vars_v_x_cur = [x_prev, y_prev, v_x_prev, v_y_prev, F_x_clip_prev, F_y_clip_prev, v_x_cur]
        network.addEquality(vars_v_x_cur, coeffs_v_x_t, 0, False)

        vars_v_y_cur = [x_prev, y_prev, v_x_prev, v_y_prev, F_x_clip_prev, F_y_clip_prev, v_y_cur]
        network.addEquality(vars_v_y_cur, coeffs_v_y_t, 0, False)
        
        
        #Property verification continues here

        #|x| for final (current) state
        x_mod_cur = network.getNewVariable()
        network.addAbsConstraint(x_cur, x_mod_cur) 

        #|y| for final (current) state
        y_mod_cur = network.getNewVariable()
        network.addAbsConstraint(y_cur, y_mod_cur) 

        #|x|+|y| for final (current) state
        xy_mod_cur = network.getNewVariable()
        network.addEquality([x_mod_cur, y_mod_cur, xy_mod_cur], [1,1,-1], 0, False)

        # USING AND OF NE NEGATED INEQUALITY
        # INTRODUCING EPSILON
        #  |v_x| + |v_y| for current state  - (|v_x| - |v_y| for prev state) >= -epsilon   
        #-(|v_x| + |v_y| for current state) + (|v_x| - |v_y| for prev state) <=  epsilon   
        network.addInequality([xy_mod_initial, xy_mod_cur], [1,-1], epsilon, False)


        #|v_x| for final (current) state
        v_x_mod_cur = network.getNewVariable()
        network.addAbsConstraint(v_x_cur, v_x_mod_cur) 

        #|v_y| for final (current) state
        v_y_mod_cur = network.getNewVariable()
        network.addAbsConstraint(v_y_cur, v_y_mod_cur) 

        #|v_x|+|v_y| for final (current) state
        v_xy_mod_cur = network.getNewVariable()
        network.addEquality([v_x_mod_cur, v_y_mod_cur, v_xy_mod_cur], [1,1,-1], 0, False)

        # USING AND OF NE NEGATED INEQUALITY
        # INTRODUCING EPSILON
        #  |v_x| + |v_y| for current state  - (|v_x| - |v_y| for prev state) >= -epsilon   
        #-(|v_x| + |v_y| for current state) + (|v_x| - |v_y| for prev state) <=  epsilon  , in other words: prev - cur <= eps
        network.addInequality([v_xy_mod_initial, v_xy_mod_cur], [1,-1], epsilon, False)
        
        prev_state = cur_state
        prev_control = network.outputVars[0][0]
        F_x_cur, F_y_cur = prev_control[0], prev_control[1]


       
    exitCode, vals, stats = network.solve(options = options, verbose = True)

    print("The least number of steps to dock = ", i+1)
    if exitCode == "unsat":
        print(" ### UNSAT. Inductive proof completed in", time.perf_counter() - start, "seconds. \n ###Break.\n")
        break
    if exitCode == "sat":
        print(" ### SAT. Counterexample found at x_cur, y_cur, v_x_cur, v_y_cur", vals[x_cur], vals[y_cur], vals[v_x_cur], vals[v_y_cur], '\n ###Trying increased N.\n')
        print("Initial state", vals[x_initial], vals[y_initial], vals[v_x_initial], vals[v_y_initial])
    else:
        print("Failed in", time.perf_counter() - start, "seconds \n")
        
