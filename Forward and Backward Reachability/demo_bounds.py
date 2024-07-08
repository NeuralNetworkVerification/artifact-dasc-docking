import math
import numpy as np
import time

from maraboupy import Marabou
from maraboupy import MarabouCore

#basic just checking for docking

start = time.perf_counter()

NUM_STEPS = 3
m = 12
n = 0.001027
t = 1

#velocity expansion is arbitrary

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

options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=30, \
    numWorkers=4, solveWithMILP=False)
networkName = "ckpt_200_with_pre_post_processing_64n.onnx"
network = Marabou.read_onnx(networkName)

# INITIALIZATION
prev_state = network.inputVars[0][0]
prev_control = network.outputVars[0][0]

x_initial = prev_state[0]
y_initial = prev_state[1]
v_x_initial = prev_state[2]
v_y_initial = prev_state[3]

# SYSTEM DYNAMICS UNROLLING
for i in range(1, NUM_STEPS):
    network.shallowClear()
    network.readONNX(networkName, None, None, reindexOutputVars=False)
    
    cur_state = network.inputVars[i][0]
    
    x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
    F_x, F_y = prev_control[0], prev_control[1]
    x_t, y_t, v_x_t, v_y_t = cur_state[0], cur_state[1], cur_state[2], cur_state[3]

    # Bind previous state to current state
    vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, x_t]
    network.addEquality(vars_x_t, coeffs_x_t, 0, False)

    vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, y_t]
    network.addEquality(vars_y_t, coeffs_y_t, 0, False)

    vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, v_x_t]
    network.addEquality(vars_v_x_t, coeffs_v_x_t, 0, False)

    vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, v_y_t]
    network.addEquality(vars_v_y_t, coeffs_v_y_t, 0, False)

    prev_state = cur_state
    prev_control = network.outputVars[0][0]
    # This needs to have first index 0 since readONNX resets

# EXTRACT FINAL STATE
x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
F_x, F_y = prev_control[0], prev_control[1]

x_t = network.getNewVariable()
y_t = network.getNewVariable()
v_x_t = network.getNewVariable()
v_y_t = network.getNewVariable()

# System dynamics for final step
vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, x_t]
network.addEquality(vars_x_t, coeffs_x_t, 0, False)

vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, y_t]
network.addEquality(vars_y_t, coeffs_y_t, 0, False)

vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, v_x_t]
network.addEquality(vars_v_x_t, coeffs_v_x_t, 0, False)

vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x, F_y, v_y_t]
network.addEquality(vars_v_y_t, coeffs_v_y_t, 0, False)


"""
Property verification begins here
"""

x_lb = 70
x_ub = 70.5
y_lb = 99
y_ub = 101.5

v_x_lb = -0.14
v_x_ub = 0.14
v_y_lb = -0.14
v_y_ub = 0.14

# INPUT BOUNDING #
network.setLowerBound(x_initial, x_lb)
network.setUpperBound(x_initial, x_ub)
network.setLowerBound(y_initial, y_lb)
network.setUpperBound(y_initial, y_ub)
network.setLowerBound(v_x_initial, v_x_lb)
network.setUpperBound(v_x_initial, v_x_ub)
network.setLowerBound(v_y_initial, v_y_lb)
network.setUpperBound(v_y_initial, v_y_ub)


k = 0.5
e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
e1.addAddend(1.0, x_t)
e1.setScalar(k)

e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
e2.addAddend(1.0, x_t)
e2.setScalar(-k)

e3 = MarabouCore.Equation(MarabouCore.Equation.GE)
e3.addAddend(1.0, y_t)
e3.setScalar(k)

e4 = MarabouCore.Equation(MarabouCore.Equation.LE)
e4.addAddend(1.0, y_t)
e4.setScalar(-k)

network.addDisjunctionConstraint([[e1], [e2], [e3], [e4]])

max_velocity = 0.2 + 2*n*(math.sqrt(max(x_ub,x_lb)**2 + max(y_ub,y_lb)**2))
print("Max velocity is: ", max_velocity)
print("Attempting docking proof for input bounds", [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]], "using", NUM_STEPS, "unroll steps")
exitCode, vals, stats = network.solve(options = options, verbose = True)
if exitCode == "sat":
    print("Counterexample found at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
if exitCode == "unsat":
    print("Docking proof completed in", time.perf_counter() - start, "seconds")
else:
    print("Failed in", time.perf_counter() - start, "seconds")
