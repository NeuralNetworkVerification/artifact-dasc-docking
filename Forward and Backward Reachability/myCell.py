"""
File: myCell.py
Adapted from file by Tobey Shim
Adaptions by Udayan Mandal

Modified cell file
"""

import math
import numpy as np
from maraboupy import Marabou
from maraboupy import MarabouCore
import time


# Clohessy-Wiltshire system dynamics parameters
m = 12  # Spacecraft mass
n = 0.001027  # Spacecraft mean motion
t = 1  # Size of discretization timetep


# Used for linear underapproximation of docking region.
# We can't represent the circular docking region using linear constraints
# (or at least not simply), so we underapproximate it with an inscribed square.
# Corners at (+- k, +- k).
k = 0.5 


# Coefficients used for system dynamics updates.
# Correspond to linearization of Clohessy-Wiltshire dynamics equations.
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

# Change as needed: a path to the controller network
networkName = "ckpt_200_with_pre_post_processing.onnx"


class Cell:
    # `bounds`: a list of lists of floats. Format:
    # [[19.0, 20.0], [19.0, 20.0], [0.3, 0.4], [0.3, 0.4]]
    # `steps`: the number of unrolling steps to perform
    # `timeout` timeout in seconds before Marabou solver should report a timeout.
    def __init__(self, bounds, steps, timeout):
        self.bounds = bounds
        self.steps = steps
        self.timeout = timeout


    # This function encodes the property of docking as a large disjunction constraint.
    # Each disjunct represents one of the possible ways of "not being inside" the docking region.
    # So, if the overall disjunction is `unsat`, that means that docking has occurred.

    # Arguments: cur_safe_region is a set of bounds (list of lists) representing the current proven region.
    # `network` is a Marabou network object corresponding to the controller.
    # `final_state` is a list of Marabou variable numbers corresponding to the 4 variables
    # representing the spacecraft's final position in the state space after all unroll steps.
    def getDisjuncts(self, network, final_state, cur_safe_region = None, inside_docking = True):
        x_t, y_t, v_x_t, v_y_t = final_state[0], final_state[1], final_state[2], final_state[3]

        # These inequalities relate to the original docking region,
        # i.e. the region which is always considered safe.
        x_dock_upper = MarabouCore.Equation(MarabouCore.Equation.GE)
        x_dock_upper.addAddend(1.0, x_t)
        x_dock_upper.setScalar(k)
        x_dock_lower = MarabouCore.Equation(MarabouCore.Equation.LE)
        x_dock_lower.addAddend(1.0, x_t)
        x_dock_lower.setScalar(-k)

        y_dock_upper = MarabouCore.Equation(MarabouCore.Equation.GE)
        y_dock_upper.addAddend(1.0, y_t)
        y_dock_upper.setScalar(k)
        y_dock_lower = MarabouCore.Equation(MarabouCore.Equation.LE)
        y_dock_lower.addAddend(1.0, y_t)
        y_dock_lower.setScalar(-k)

        if not inside_docking:  
            # These inequalities relate to the *proven* safe region,
            # which will include regions outside the docking region
            # which have previously been proven.
            x_min, y_min, v_x_min, v_y_min = cur_safe_region[0][0], cur_safe_region[1][0], cur_safe_region[2][0], cur_safe_region[3][0]
            x_max, y_max, v_x_max, v_y_max = cur_safe_region[0][1], cur_safe_region[1][1], cur_safe_region[2][1], cur_safe_region[3][1]

            x_safe_upper = MarabouCore.Equation(MarabouCore.Equation.GE)
            x_safe_upper.addAddend(1.0, x_t)
            x_safe_upper.setScalar(x_max)

            x_safe_lower = MarabouCore.Equation(MarabouCore.Equation.LE)
            x_safe_lower.addAddend(1.0, x_t)
            x_safe_lower.setScalar(x_min)

            y_safe_upper = MarabouCore.Equation(MarabouCore.Equation.GE)
            y_safe_upper.addAddend(1.0, y_t)
            y_safe_upper.setScalar(y_max)

            y_safe_lower = MarabouCore.Equation(MarabouCore.Equation.LE)
            y_safe_lower.addAddend(1.0, y_t)
            y_safe_lower.setScalar(y_min)

            v_x_safe_upper = MarabouCore.Equation(MarabouCore.Equation.GE)
            v_x_safe_upper.addAddend(1.0, v_x_t)
            v_x_safe_upper.setScalar(v_x_max)  

            v_x_safe_lower = MarabouCore.Equation(MarabouCore.Equation.LE)
            v_x_safe_lower.addAddend(1.0, v_x_t)
            v_x_safe_lower.setScalar(v_x_min)

            v_y_safe_upper = MarabouCore.Equation(MarabouCore.Equation.GE)
            v_y_safe_upper.addAddend(1.0, v_y_t)
            v_y_safe_upper.setScalar(v_y_max)

            v_y_safe_lower = MarabouCore.Equation(MarabouCore.Equation.LE)
            v_y_safe_lower.addAddend(1.0, v_y_t)
            v_y_safe_lower.setScalar(v_y_min)    

            network.addDisjunctionConstraint(
                [
                    # Outside the safe region altogether.
                    [x_safe_lower], [x_safe_upper], [y_safe_lower], [y_safe_upper],

                    # You're both not inside the docking region
                    # and at a velocity which hasn't been proven yet.
                    [v_x_safe_lower, x_dock_lower], [v_x_safe_lower, x_dock_upper], [v_x_safe_upper, x_dock_lower], [v_x_safe_upper, x_dock_upper],
                    [v_x_safe_lower, y_dock_lower], [v_x_safe_lower, y_dock_upper], [v_x_safe_upper, y_dock_lower], [v_x_safe_upper, y_dock_upper],
                    [v_y_safe_lower, x_dock_lower], [v_y_safe_lower, x_dock_upper], [v_y_safe_upper, x_dock_lower], [v_y_safe_upper, x_dock_upper],
                    [v_y_safe_lower, y_dock_lower], [v_y_safe_lower, y_dock_upper], [v_y_safe_upper, y_dock_lower], [v_y_safe_upper, y_dock_upper]
                ]
            )
        else:
            network.addDisjunctionConstraint(
                [
                    # Outside the safe region altogether.
                    [x_dock_lower], [x_dock_upper], [y_dock_lower], [y_dock_upper]
                ]
            )




    # Unrolls the specified cell for self.steps many iterations & checks docking at the end.
    # cur_safe_region is a list of 4 lists representing the current bounds.
    def unroll(self, cur_safe_region = None, inside_docking=True):
        # INITIALIZATION
        network = Marabou.read_onnx(networkName)
        prev_state = network.inputVars[0][0]
        prev_control = network.outputVars[0][0]
        x_initial = prev_state[0]
        y_initial = prev_state[1]
        v_x_initial = prev_state[2]
        v_y_initial = prev_state[3]

        # SYSTEM DYNAMICS UNROLLING
        for i in range(1, self.steps):
            # Marabou needs to be modified for this to work.
            # See README.
            network.shallowClear()
            network.readONNX(networkName, None, None, reindexOutputVars=False)
    
            cur_state = network.inputVars[i][0]
    
            x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
            F_x, F_y = prev_control[0], prev_control[1]
            x_t, y_t, v_x_t, v_y_t = cur_state[0], cur_state[1], cur_state[2], cur_state[3]

            # Bind previous state to current state
            # Use Marabou's equality constraint to directly encode system dynamics
            # as a "constraint".
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
            
        # Extract final state.
        x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
        F_x, F_y = prev_control[0], prev_control[1]

        # Network variables representing final state.
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

        # Set initial state bounds (input bounds).
        network.setLowerBound(x_initial, self.bounds[0][0])
        network.setUpperBound(x_initial, self.bounds[0][1])
        network.setLowerBound(y_initial, self.bounds[1][0])
        network.setUpperBound(y_initial, self.bounds[1][1])
        network.setLowerBound(v_x_initial, self.bounds[2][0])
        network.setUpperBound(v_x_initial, self.bounds[2][1])
        network.setLowerBound(v_y_initial, self.bounds[3][0])
        network.setUpperBound(v_y_initial, self.bounds[3][1])

        # Add resulting disjuncts to network.
        self.getDisjuncts(network, [x_t, y_t, v_x_t, v_y_t], cur_safe_region, inside_docking)
        
        # Solver output.
        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=self.timeout, \
        numWorkers=4, solveWithMILP=False)
        start_time = time.perf_counter()
        exitCode, vals, stats = network.solve(options = options, verbose=True)
        time_taken = time.perf_counter() - start_time
        #return exitCode == "unsat"
        if exitCode == "sat":
            print("Counter example found at: ", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
        return exitCode, time_taken
