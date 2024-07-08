import numpy as np
import time

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
from maraboupy import Marabou
from maraboupy import MarabouCore
# from maraboupy import MarabouCore
class Queries:

    def __init__(self):

        #self network name
        self.networkName = "ckpt_200_with_pre_post_processing_4safe.onnx"

        #fixed m,n,t
        self.m = 12
        self.n = 0.001027
        self.t = 1

        # Matrix encoding of system dynamics
        self.coeffs_x_t = [
            4 - 3 * np.cos(self.n * self.t),
            0,
            1 / self.n * np.sin(self.n * self.t),
            2 / self.n - 2 / self.n * np.cos(self.n * self.t),
            (1 - np.cos(self.n * self.t)) / (self.m * self.n ** 2),
            2 * self.t / (self.m * self.n) - 2 * np.sin(self.n * self.t) / (self.m * self.n ** 2),
            -1
        ]
        self.coeffs_y_t = [
            -6 * self.n * self.t + 6 * np.sin(self.n * self.t),
            1,
            -2 / self.n + 2 / self.n * np.cos(self.n * self.t),
            -3 * self.t + 4 / self.n * np.sin(self.n * self.t),
            (-2 * self.t) / (self.m * self.n) + (2 * np.sin(self.n * self.t)) / (self.m * self.n ** 2),
            4 / (self.m * self.n ** 2) - (3 * self.t ** 2) / (2 * self.m) - (4 * np.cos(self.n * self.t)) / (self.m * self.n ** 2),
            -1
        ]
        self.coeffs_v_x_t = [
            3 * self.n * np.sin(self.n * self.t),
            0,
            np.cos(self.n * self.t),
            2 * np.sin(self.n * self.t),
            np.sin(self.n * self.t) / (self.m * self.n),
            2 / (self.m * self.n) - (2 * np.cos(self.n * self.t)) / (self.m * self.n),
            -1
        ]
        self.coeffs_v_y_t = [
            -6 * self.n + 6 * self.n * np.cos(self.n * self.t),
            0,
            -2 * np.sin(self.n * self.t),
            -3 + 4 * np.cos(self.n * self.t),
            (2 * np.cos(self.n * self.t) - 2) / (self.m * self.n),
            (-3 * self.t) / (self.m) + (4 * np.sin(self.n * self.t)) / (self.m * self.n),
            -1
        ]
    '''
    check if current cell can reach target cell in 1 step
    '''
    def transitioncheck(self, current_cell, target_cell, timeout, numsteps):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = True

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        x_0, y_0, v_x_0, v_y_0, x_t, y_t, v_x_t, v_y_t = self.run_unrolls(network, numsteps)

        #now run the query
        x_lb = current_cell[0][0]
        x_ub = current_cell[0][1]
        y_lb = current_cell[1][0]
        y_ub = current_cell[1][1]

        v_x_lb = current_cell[2][0]
        v_x_ub = current_cell[2][1]
        v_y_lb = current_cell[3][0]
        v_y_ub = current_cell[3][1]

        # INPUT BOUNDING #
        network.setLowerBound(x_0, x_lb)
        network.setUpperBound(x_0, x_ub)
        network.setLowerBound(y_0, y_lb)
        network.setUpperBound(y_0, y_ub)
        network.setLowerBound(v_x_0, v_x_lb)
        network.setUpperBound(v_x_0, v_x_ub)
        network.setLowerBound(v_y_0, v_y_lb)
        network.setUpperBound(v_y_0, v_y_ub)

        #check for ending of in the resulting cell
        transition_x_lb = target_cell[0][0]
        transition_x_ub = target_cell[0][1]

        transition_y_lb = target_cell[1][0]
        transition_y_ub = target_cell[1][1]

        transition_vx_lb = target_cell[2][0]
        transition_vx_ub = target_cell[2][1]

        transition_vy_lb = target_cell[3][0]
        transition_vy_ub = target_cell[3][1]

        e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e1.addAddend(1.0, x_t)
        e1.setScalar(1.0 * transition_x_ub)

        e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e2.addAddend(1.0, x_t)
        e2.setScalar(1.0 * transition_x_lb)

        e3 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e3.addAddend(1.0, y_t)
        e3.setScalar(1.0 * transition_y_ub)

        e4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e4.addAddend(1.0, y_t)
        e4.setScalar(1.0 * transition_y_lb)

        e5 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e5.addAddend(1.0, v_x_t)
        e5.setScalar(1.0 * transition_vx_ub)

        e6 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e6.addAddend(1.0, v_x_t)
        e6.setScalar(1.0 * transition_vx_lb)

        e7 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e7.addAddend(1.0, v_y_t)
        e7.setScalar(1.0 * transition_vy_ub)

        e8 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e8.addAddend(1.0, v_y_t)
        e8.setScalar(1.0 * transition_vy_lb)
        network.addDisjunctionConstraint([[e1, e2, e3, e4, e5, e6, e7, e8]])

        # network.addDisjunctionConstraint([[e3]])

        print("Attempting proof for input bounds",
              [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]], "using", numsteps, "unroll steps")
        print("Attempting to reach bounds",
              [[transition_x_lb, transition_x_ub], [transition_y_lb, transition_y_ub], [transition_vx_lb, transition_vx_ub], [transition_vy_lb, transition_vy_ub]])

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return True
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return False
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return False

    def always_reaches(self, current_cell, target_cell, timeout, numsteps):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = True

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        x_0, y_0, v_x_0, v_y_0, x_t, y_t, v_x_t, v_y_t = self.run_unrolls(network, numsteps)

        #now run the query
        x_lb = current_cell[0][0]
        x_ub = current_cell[0][1]
        y_lb = current_cell[1][0]
        y_ub = current_cell[1][1]

        v_x_lb = current_cell[2][0]
        v_x_ub = current_cell[2][1]
        v_y_lb = current_cell[3][0]
        v_y_ub = current_cell[3][1]

        # INPUT BOUNDING #
        network.setLowerBound(x_0, x_lb)
        network.setUpperBound(x_0, x_ub)
        network.setLowerBound(y_0, y_lb)
        network.setUpperBound(y_0, y_ub)
        network.setLowerBound(v_x_0, v_x_lb)
        network.setUpperBound(v_x_0, v_x_ub)
        network.setLowerBound(v_y_0, v_y_lb)
        network.setUpperBound(v_y_0, v_y_ub)

        #check for ending of in the resulting cell
        transition_x_lb = target_cell[0][0]
        transition_x_ub = target_cell[0][1]

        transition_y_lb = target_cell[1][0]
        transition_y_ub = target_cell[1][1]

        transition_vx_lb = target_cell[2][0]
        transition_vx_ub = target_cell[2][1]

        transition_vy_lb = target_cell[3][0]
        transition_vy_ub = target_cell[3][1]

        e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e1.addAddend(1.0, x_t)
        e1.setScalar(1.0 * transition_x_ub)

        e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e2.addAddend(1.0, x_t)
        e2.setScalar(1.0 * transition_x_lb)

        e3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e3.addAddend(1.0, y_t)
        e3.setScalar(1.0 * transition_y_ub)

        e4 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e4.addAddend(1.0, y_t)
        e4.setScalar(1.0 * transition_y_lb)

        e5 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e5.addAddend(1.0, v_x_t)
        e5.setScalar(1.0 * transition_vx_ub)

        e6 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e6.addAddend(1.0, v_x_t)
        e6.setScalar(1.0 * transition_vx_lb)

        e7 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e7.addAddend(1.0, v_y_t)
        e7.setScalar(1.0 * transition_vy_ub)

        e8 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e8.addAddend(1.0, v_y_t)
        e8.setScalar(1.0 * transition_vy_lb)
        network.addDisjunctionConstraint([[e1], [e2], [e3], [e4], [e5], [e6], [e7], [e8]])

        # network.addDisjunctionConstraint([[e3]])

        print("Attempting proof for input bounds",
              [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]], "using", numsteps, "unroll steps")
        print("Attempting to reach bounds",
              [[transition_x_lb, transition_x_ub], [transition_y_lb, transition_y_ub], [transition_vx_lb, transition_vx_ub], [transition_vy_lb, transition_vy_ub]])

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return False
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return True
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return False


    def query_bounds(self, input, coordinate, bound, timeout, numsteps):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = True

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        x_0, y_0, v_x_0, v_y_0, x_t, y_t, v_x_t, v_y_t = self.run_unrolls(network, numsteps)

        #now run the query
        x_lb = input[0][0]
        x_ub = input[0][1]
        y_lb = input[1][0]
        y_ub = input[1][1]

        v_x_lb = input[2][0]
        v_x_ub = input[2][1]
        v_y_lb = input[3][0]
        v_y_ub = input[3][1]

        # INPUT BOUNDING #
        network.setLowerBound(x_0, x_lb)
        network.setUpperBound(x_0, x_ub)
        network.setLowerBound(y_0, y_lb)
        network.setUpperBound(y_0, y_ub)
        network.setLowerBound(v_x_0, v_x_lb)
        network.setUpperBound(v_x_0, v_x_ub)
        network.setLowerBound(v_y_0, v_y_lb)
        network.setUpperBound(v_y_0, v_y_ub)


        #OUTPUT BOUNDING
        if coordinate == 'x':
            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, x_t)
            e1.addAddend(-1.0, x_0)
            e1.setScalar(bound)

            e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e2.addAddend(1.0, x_0)
            e2.addAddend(-1.0, x_t)
            e2.setScalar(bound)

        elif coordinate == 'y':
            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, y_t)
            e1.addAddend(-1.0, y_0)
            e1.setScalar(bound)

            e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e2.addAddend(1.0, y_0)
            e2.addAddend(-1.0, y_t)
            e2.setScalar(bound)

        elif coordinate == 'v_x':
            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, v_x_t)
            e1.addAddend(-1.0, v_x_0)
            e1.setScalar(bound)

            e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e2.addAddend(1.0, v_x_0)
            e2.addAddend(-1.0, v_x_t)
            e2.setScalar(bound)

        elif coordinate == 'v_y':
            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, v_y_t)
            e1.addAddend(-1.0, v_y_0)
            e1.setScalar(bound)

            e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e2.addAddend(1.0, v_y_0)
            e2.addAddend(-1.0, v_y_t)
            e2.setScalar(bound)

        e3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e3.addAddend(1.0, x_0)
        e3.setScalar(0.5)

        e4 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e4.addAddend(1.0, x_0)
        e4.setScalar(-0.5)

        e5 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e5.addAddend(1.0, y_0)
        e5.setScalar(0.5)

        e6 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e6.addAddend(1.0, y_0)
        e6.setScalar(-0.5)

        network.addDisjunctionConstraint([[e3],[e4],[e5],[e6]])
        network.addDisjunctionConstraint([[e1], [e2]])

        print("Attempting proof for input bounds",
              [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]], "using", numsteps, "unroll steps")
        print("Attempting to prove bound",
              bound)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return 0
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return 1
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return -1

    def check_fine_velocity(self, input, bound, timeout, numsteps):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = True

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        x_0, y_0, v_x_0, v_y_0, x_t, y_t, v_x_t, v_y_t = self.run_unrolls(network, numsteps)

        #now run the query
        x_lb = input[0][0]
        x_ub = input[0][1]
        y_lb = input[1][0]
        y_ub = input[1][1]

        v_x_lb = -bound
        v_x_ub = bound
        v_y_lb = -bound
        v_y_ub = bound

        # INPUT BOUNDING #
        network.setLowerBound(x_0, x_lb)
        network.setUpperBound(x_0, x_ub)
        network.setLowerBound(y_0, y_lb)
        network.setUpperBound(y_0, y_ub)
        network.setLowerBound(v_x_0, v_x_lb)
        network.setUpperBound(v_x_0, v_x_ub)
        network.setLowerBound(v_y_0, v_y_lb)
        network.setUpperBound(v_y_0, v_y_ub)


        #OUTPUT BOUNDING
        '''
        e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e1.addAddend(1.0, v_x_t)
        e1.addAddend(-1.0, v_x_0)
        e1.setScalar(-bound)

        e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e2.addAddend(1.0, v_x_0)
        e2.addAddend(-1.0, v_x_t)
        e2.setScalar(-bound)

        e3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e3.addAddend(1.0, v_y_t)
        e3.addAddend(-1.0, v_y_0)
        e3.setScalar(-bound)

        e4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e4.addAddend(1.0, v_y_0)
        e4.addAddend(-1.0, v_y_t)
        e4.setScalar(-bound)
        '''
        e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e1.addAddend(1.0, v_x_t)
        e1.setScalar(-bound)

        e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e2.addAddend(1.0, v_x_t)
        e2.setScalar(bound)

        e3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e3.addAddend(1.0, v_y_t)
        e3.setScalar(-bound)

        e4 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e4.addAddend(1.0, v_y_t)
        e4.setScalar(bound)
        '''
        e5 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e5.addAddend(1.0, x_t)
        e5.addAddend(-1.0, x_0)
        e5.setScalar(-bound)

        e6 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e6.addAddend(1.0, x_0)
        e6.addAddend(-1.0, x_t)
        e6.setScalar(-bound)

        e7 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e7.addAddend(1.0, y_t)
        e7.addAddend(-1.0, y_0)
        e7.setScalar(-bound)

        e8 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e8.addAddend(1.0, y_0)
        e8.addAddend(-1.0, y_t)
        e8.setScalar(-bound)
        '''

        e5 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e5.addAddend(1.0, x_0)
        e5.setScalar(0.5)

        e6 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e6.addAddend(1.0, x_0)
        e6.setScalar(-0.5)

        e7 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e7.addAddend(1.0, y_0)
        e7.setScalar(0.5)

        e8 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e8.addAddend(1.0, y_0)
        e8.setScalar(-0.5)

        network.addDisjunctionConstraint([[e5],[e6],[e7],[e8]])

        #network.addDisjunctionConstraint([[e1,e2,e3,e4,e5,e6,e7,e8]])
        network.addDisjunctionConstraint([[e1, e2, e3, e4]])

        print("Attempting proof for input bounds",
              [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]], "using", numsteps, "unroll steps")
        print("Attempting to prove bound",
              bound)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return 0
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return 1
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return -1


    def check_cycle(self, input, first, bound, timeout):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = True

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        prev_state = network.inputVars[0][0]
        prev_control = network.outputVars[0][0]

        x_initial = prev_state[0]
        y_initial = prev_state[1]
        v_x_initial = prev_state[2]
        v_y_initial = prev_state[3]

        clip_lb = -1
        clip_ub = 1

        # SYSTEM DYNAMICS UNROLLING
        for i in range(1):
            network.shallowClear()
            network.readONNX(self.networkName, None, None, reindexOutputVars=False)

            cur_state = network.inputVars[i][0]  # the state at t=i for the network

            x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]  # t=i-1
            F_x, F_y = prev_control[0], prev_control[1]
            x_t, y_t, v_x_t, v_y_t = cur_state[0], cur_state[1], cur_state[2], cur_state[3]

            # handling F_x_clipping -- TODO: add method in Marabou for this
            aux1 = network.getNewVariable()
            F_x_clip = network.getNewVariable()
            network.addEquality([aux1, F_x], [1, -1], -1 * clip_lb, False)  # aux1 = F_x-clip_lb
            aux2 = network.getNewVariable()
            network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
            aux3 = network.getNewVariable()
            network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
            aux4 = network.getNewVariable()
            network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
            aux5 = network.getNewVariable()
            network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
            network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)  # F_x_clip = clip_ub - aux5

            # handling F_y_clipping
            aux1 = network.getNewVariable()
            F_y_clip = network.getNewVariable()
            network.addEquality([aux1, F_y], [1, -1], -1 * clip_lb, False)  # aux1 = F_y-clip_lb
            aux2 = network.getNewVariable()
            network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
            aux3 = network.getNewVariable()
            network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
            aux4 = network.getNewVariable()
            network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
            aux5 = network.getNewVariable()
            network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
            network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)  # F_y_clip = clip_ub - aux5

            # Bind previous state to current state
            vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, x_t]
            network.addEquality(vars_x_t, self.coeffs_x_t, 0, False)

            vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, y_t]
            network.addEquality(vars_y_t, self.coeffs_y_t, 0, False)

            vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_x_t]
            network.addEquality(vars_v_x_t, self.coeffs_v_x_t, 0, False)

            vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_y_t]
            network.addEquality(vars_v_y_t, self.coeffs_v_y_t, 0, False)

            prev_state = cur_state
            prev_control = network.outputVars[0][0]  # pretty confident this should be i
            # This needs to have first index 0 since readONNX resets

            # at each step, use current state and current output (output from passing in current state) to get next state
            # t0 --prev state, prev control (state at t0, output from network passing in state at t0)
            # t1 --from closed form map prev state, prev control (state at t0, output at state at t0) to state at t1
            # loops until steps - 1 -- at loop termination we have prev state, prev control (state at t-1, output from network passing in state at t-1)

        x_1, y_1, v_x_1, v_y_1 = x_t, y_t, v_x_t, v_y_t
        # EXTRACT FINAL STATE
        x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
        F_x, F_y = prev_control[0], prev_control[1]

        # handling F_x_clipping -- TODO: add method in Marabou for this
        aux1 = network.getNewVariable()
        F_x_clip = network.getNewVariable()
        network.addEquality([aux1, F_x], [1, -1], -1 * clip_lb, False)  # aux1 = F_x-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)  # F_x_clip = clip_ub - aux5

        # handling F_y_clipping
        aux1 = network.getNewVariable()
        F_y_clip = network.getNewVariable()
        network.addEquality([aux1, F_y], [1, -1], -1 * clip_lb, False)  # aux1 = F_y-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)  # F_y_clip = clip_ub - aux5

        x_t = network.getNewVariable()
        y_t = network.getNewVariable()
        v_x_t = network.getNewVariable()
        v_y_t = network.getNewVariable()

        # System dynamics for final step
        vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, x_t]
        network.addEquality(vars_x_t, self.coeffs_x_t, 0, False)

        vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, y_t]
        network.addEquality(vars_y_t, self.coeffs_y_t, 0, False)

        vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_x_t]
        network.addEquality(vars_v_x_t, self.coeffs_v_x_t, 0, False)

        vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_y_t]
        network.addEquality(vars_v_y_t, self.coeffs_v_y_t, 0, False)

        x_0, y_0, v_x_0, v_y_0 = x_initial, y_initial, v_x_initial, v_y_initial

        #now run the query
        x_lb = input[0][0]
        x_ub = input[0][1]
        y_lb = input[1][0]
        y_ub = input[1][1]

        v_x_lb = input[2][0]
        v_x_ub = input[2][1]
        v_y_lb = input[3][0]
        v_y_ub = input[3][1]

        # INPUT BOUNDING #
        network.setLowerBound(x_0, x_lb)
        network.setUpperBound(x_0, x_ub)
        network.setLowerBound(y_0, y_lb)
        network.setUpperBound(y_0, y_ub)
        if first == 'negative':
            network.setUpperBound(v_x_0, -bound)
            network.setUpperBound(v_y_0, -bound)
        else:
            network.setLowerBound(v_x_0, bound)
            network.setLowerBound(v_y_0, bound)


        #OUTPUT BOUNDING

        e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e1.addAddend(1.0, v_x_1)
        e1.setScalar(bound)

        e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e2.addAddend(1.0, v_x_1)
        e2.setScalar(-bound)

        e3 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e3.addAddend(1.0, v_y_1)
        e3.setScalar(bound)

        e4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e4.addAddend(1.0, v_y_1)
        e4.setScalar(-bound)

        if first == 'negative':
            e5 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e5.addAddend(1.0, v_x_t)
            e5.setScalar(bound)

            e6 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e6.addAddend(1.0, v_y_t)
            e6.setScalar(bound)

        else:
            e5 = MarabouCore.Equation(MarabouCore.Equation.LE)
            e5.addAddend(1.0, v_x_t)
            e5.setScalar(-bound)

            e6 = MarabouCore.Equation(MarabouCore.Equation.LE)
            e6.addAddend(1.0, v_y_t)
            e6.setScalar(-bound)

        #network.addDisjunctionConstraint([[e1, e2, e3, e4, e5, e6]])

        print("Attempting proof for input bounds",
              [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]])
        print("Attempting to prove bound",
              bound)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return 0
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return 1
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return -1

    def check_velocity_range(self, vel, input, timeout):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solve with MILP var

        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = True

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        x_0, y_0, v_x_0, v_y_0, x_t, y_t, v_x_t, v_y_t = self.run_unrolls(network, 1)

        #now run the query
        x_lb = input[0][0]
        x_ub = input[0][1]
        y_lb = input[1][0]
        y_ub = input[1][1]

        v_x_lb = -vel
        v_x_ub = vel
        v_y_lb = -vel
        v_y_ub = vel

        # INPUT BOUNDING #
        network.setLowerBound(x_0, x_lb)
        network.setUpperBound(x_0, x_ub)
        network.setLowerBound(y_0, y_lb)
        network.setUpperBound(y_0, y_ub)
        network.setLowerBound(v_x_0, v_x_lb)
        network.setUpperBound(v_x_0, v_x_ub)
        network.setLowerBound(v_y_0, v_y_lb)
        network.setUpperBound(v_y_0, v_y_ub)

        e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e1.addAddend(1.0, v_x_t)
        e1.setScalar(-vel)

        e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e2.addAddend(1.0, v_x_t)
        e2.setScalar(vel)

        e3 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e3.addAddend(1.0, v_y_t)
        e3.setScalar(-vel)

        e4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e4.addAddend(1.0, v_y_t)
        e4.setScalar(vel)

        network.addDisjunctionConstraint([[e1],[e2],[e3],[e4]])

        print("Attempting proof for input bounds",
              [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]])

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return 0
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return 1
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return -1


    def check_exists_seq(self, sequence, start_ind, end_ind, timeout):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = True

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        prev_state = network.inputVars[0][0]
        prev_control = network.outputVars[0][0]

        x_initial = prev_state[0]
        y_initial = prev_state[1]
        v_x_initial = prev_state[2]
        v_y_initial = prev_state[3]

        clip_lb = -1
        clip_ub = 1

        arr = []

        # SYSTEM DYNAMICS UNROLLING
        for i in range(1, end_ind - start_ind):
            network.shallowClear()
            network.readONNX(self.networkName, None, None, reindexOutputVars=False)

            cur_state = network.inputVars[i][0]  # the state at t=i for the network

            x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]  # t=i-1
            F_x, F_y = prev_control[0], prev_control[1]
            x_t, y_t, v_x_t, v_y_t = cur_state[0], cur_state[1], cur_state[2], cur_state[3]

            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, x_t)
            e1.setScalar(sequence[start_ind + i][0][0])
            arr.append(e1)

            e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
            e2.addAddend(1.0, x_t)
            e2.setScalar(sequence[start_ind + i][0][1])
            arr.append(e2)

            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, y_t)
            e1.setScalar(sequence[start_ind + i][1][0])
            arr.append(e1)

            e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
            e2.addAddend(1.0, y_t)
            e2.setScalar(sequence[start_ind + i][1][1])
            arr.append(e2)

            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, v_x_t)
            e1.setScalar(sequence[start_ind + i][2][0])
            arr.append(e1)

            e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
            e2.addAddend(1.0, v_x_t)
            e2.setScalar(sequence[start_ind + i][2][1])
            arr.append(e2)

            e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            e1.addAddend(1.0, v_y_t)
            e1.setScalar(sequence[start_ind + i][3][0])
            arr.append(e1)

            e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
            e2.addAddend(1.0, v_y_t)
            e2.setScalar(sequence[start_ind + i][3][1])
            arr.append(e2)

            # handling F_x_clipping -- TODO: add method in Marabou for this
            aux1 = network.getNewVariable()
            F_x_clip = network.getNewVariable()
            network.addEquality([aux1, F_x], [1, -1], -1 * clip_lb, False)  # aux1 = F_x-clip_lb
            aux2 = network.getNewVariable()
            network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
            aux3 = network.getNewVariable()
            network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
            aux4 = network.getNewVariable()
            network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
            aux5 = network.getNewVariable()
            network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
            network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)  # F_x_clip = clip_ub - aux5

            # handling F_y_clipping
            aux1 = network.getNewVariable()
            F_y_clip = network.getNewVariable()
            network.addEquality([aux1, F_y], [1, -1], -1 * clip_lb, False)  # aux1 = F_y-clip_lb
            aux2 = network.getNewVariable()
            network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
            aux3 = network.getNewVariable()
            network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
            aux4 = network.getNewVariable()
            network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
            aux5 = network.getNewVariable()
            network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
            network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)  # F_y_clip = clip_ub - aux5

            # Bind previous state to current state
            vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, x_t]
            network.addEquality(vars_x_t, self.coeffs_x_t, 0, False)

            vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, y_t]
            network.addEquality(vars_y_t, self.coeffs_y_t, 0, False)

            vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_x_t]
            network.addEquality(vars_v_x_t, self.coeffs_v_x_t, 0, False)

            vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_y_t]
            network.addEquality(vars_v_y_t, self.coeffs_v_y_t, 0, False)

            prev_state = cur_state
            prev_control = network.outputVars[0][0]  # pretty confident this should be i
            # This needs to have first index 0 since readONNX resets

            # at each step, use current state and current output (output from passing in current state) to get next state
            # t0 --prev state, prev control (state at t0, output from network passing in state at t0)
            # t1 --from closed form map prev state, prev control (state at t0, output at state at t0) to state at t1
            # loops until steps - 1 -- at loop termination we have prev state, prev control (state at t-1, output from network passing in state at t-1)

        # EXTRACT FINAL STATE
        x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
        F_x, F_y = prev_control[0], prev_control[1]

        # handling F_x_clipping -- TODO: add method in Marabou for this
        aux1 = network.getNewVariable()
        F_x_clip = network.getNewVariable()
        network.addEquality([aux1, F_x], [1, -1], -1 * clip_lb, False)  # aux1 = F_x-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)  # F_x_clip = clip_ub - aux5

        # handling F_y_clipping
        aux1 = network.getNewVariable()
        F_y_clip = network.getNewVariable()
        network.addEquality([aux1, F_y], [1, -1], -1 * clip_lb, False)  # aux1 = F_y-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)  # F_y_clip = clip_ub - aux5

        x_t = network.getNewVariable()
        y_t = network.getNewVariable()
        v_x_t = network.getNewVariable()
        v_y_t = network.getNewVariable()

        # System dynamics for final step
        vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, x_t]
        network.addEquality(vars_x_t, self.coeffs_x_t, 0, False)

        vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, y_t]
        network.addEquality(vars_y_t, self.coeffs_y_t, 0, False)

        vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_x_t]
        network.addEquality(vars_v_x_t, self.coeffs_v_x_t, 0, False)

        vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_y_t]
        network.addEquality(vars_v_y_t, self.coeffs_v_y_t, 0, False)

        x_0, y_0, v_x_0, v_y_0 = x_initial, y_initial, v_x_initial, v_y_initial

        #now run the query

        # INPUT BOUNDING #
        network.setLowerBound(x_0, sequence[start_ind][0][0])
        network.setUpperBound(x_0, sequence[start_ind][0][1])
        network.setLowerBound(y_0, sequence[start_ind][1][0])
        network.setUpperBound(y_0, sequence[start_ind][1][1])
        network.setLowerBound(v_x_0, sequence[start_ind][2][0])
        network.setUpperBound(v_x_0, sequence[start_ind][2][1])
        network.setLowerBound(v_y_0, sequence[start_ind][3][0])
        network.setUpperBound(v_y_0, sequence[start_ind][3][1])


        #OUTPUT BOUNDING

        e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e1.addAddend(1.0, x_t)
        e1.setScalar(sequence[end_ind][0][0])
        arr.append(e1)

        e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e2.addAddend(1.0, x_t)
        e2.setScalar(sequence[end_ind][0][1])
        arr.append(e2)

        e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e1.addAddend(1.0, y_t)
        e1.setScalar(sequence[end_ind][1][0])
        arr.append(e1)

        e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e2.addAddend(1.0, y_t)
        e2.setScalar(sequence[end_ind][1][1])
        arr.append(e2)

        e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e1.addAddend(1.0, v_x_t)
        e1.setScalar(sequence[end_ind][2][0])
        arr.append(e1)

        e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e2.addAddend(1.0, v_x_t)
        e2.setScalar(sequence[end_ind][2][1])
        arr.append(e2)

        e1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e1.addAddend(1.0, v_y_t)
        e1.setScalar(sequence[end_ind][3][0])
        arr.append(e1)

        e2 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e2.addAddend(1.0, v_y_t)
        e2.setScalar(sequence[end_ind][3][1])
        arr.append(e2)

        network.addDisjunctionConstraint([arr])

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return 0
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return 1
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return -1

    def run_unrolls(self, network, numsteps):
        # INITIALIZATION
        prev_state = network.inputVars[0][0]
        prev_control = network.outputVars[0][0]

        x_initial = prev_state[0]
        y_initial = prev_state[1]
        v_x_initial = prev_state[2]
        v_y_initial = prev_state[3]

        clip_lb = -1
        clip_ub = 1

        # SYSTEM DYNAMICS UNROLLING
        for i in range(1, numsteps):
            network.shallowClear()
            network.readONNX(self.networkName, None, None, reindexOutputVars=False)

            cur_state = network.inputVars[i][0]  # the state at t=i for the network

            x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]  # t=i-1
            F_x, F_y = prev_control[0], prev_control[1]
            x_t, y_t, v_x_t, v_y_t = cur_state[0], cur_state[1], cur_state[2], cur_state[3]

            # handling F_x_clipping -- TODO: add method in Marabou for this
            aux1 = network.getNewVariable()
            F_x_clip = network.getNewVariable()
            network.addEquality([aux1, F_x], [1, -1], -1 * clip_lb, False)  # aux1 = F_x-clip_lb
            aux2 = network.getNewVariable()
            network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
            aux3 = network.getNewVariable()
            network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
            aux4 = network.getNewVariable()
            network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
            aux5 = network.getNewVariable()
            network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
            network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)  # F_x_clip = clip_ub - aux5

            # handling F_y_clipping
            aux1 = network.getNewVariable()
            F_y_clip = network.getNewVariable()
            network.addEquality([aux1, F_y], [1, -1], -1 * clip_lb, False)  # aux1 = F_y-clip_lb
            aux2 = network.getNewVariable()
            network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
            aux3 = network.getNewVariable()
            network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
            aux4 = network.getNewVariable()
            network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
            aux5 = network.getNewVariable()
            network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
            network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)  # F_y_clip = clip_ub - aux5

            # Bind previous state to current state
            vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, x_t]
            network.addEquality(vars_x_t, self.coeffs_x_t, 0, False)

            vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, y_t]
            network.addEquality(vars_y_t, self.coeffs_y_t, 0, False)

            vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_x_t]
            network.addEquality(vars_v_x_t, self.coeffs_v_x_t, 0, False)

            vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_y_t]
            network.addEquality(vars_v_y_t, self.coeffs_v_y_t, 0, False)

            prev_state = cur_state
            prev_control = network.outputVars[0][0]  # pretty confident this should be i
            # This needs to have first index 0 since readONNX resets

            # at each step, use current state and current output (output from passing in current state) to get next state
            # t0 --prev state, prev control (state at t0, output from network passing in state at t0)
            # t1 --from closed form map prev state, prev control (state at t0, output at state at t0) to state at t1
            # loops until steps - 1 -- at loop termination we have prev state, prev control (state at t-1, output from network passing in state at t-1)

        # EXTRACT FINAL STATE
        x_0, y_0, v_x_0, v_y_0 = prev_state[0], prev_state[1], prev_state[2], prev_state[3]
        F_x, F_y = prev_control[0], prev_control[1]

        # handling F_x_clipping -- TODO: add method in Marabou for this
        aux1 = network.getNewVariable()
        F_x_clip = network.getNewVariable()
        network.addEquality([aux1, F_x], [1, -1], -1 * clip_lb, False)  # aux1 = F_x-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)  # F_x_clip = clip_ub - aux5

        # handling F_y_clipping
        aux1 = network.getNewVariable()
        F_y_clip = network.getNewVariable()
        network.addEquality([aux1, F_y], [1, -1], -1 * clip_lb, False)  # aux1 = F_y-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)  # F_y_clip = clip_ub - aux5

        x_t = network.getNewVariable()
        y_t = network.getNewVariable()
        v_x_t = network.getNewVariable()
        v_y_t = network.getNewVariable()

        # System dynamics for final step
        vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, x_t]
        network.addEquality(vars_x_t, self.coeffs_x_t, 0, False)

        vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, y_t]
        network.addEquality(vars_y_t, self.coeffs_y_t, 0, False)

        vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_x_t]
        network.addEquality(vars_v_x_t, self.coeffs_v_x_t, 0, False)

        vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_y_t]
        network.addEquality(vars_v_y_t, self.coeffs_v_y_t, 0, False)

        return x_initial, y_initial, v_x_initial, v_y_initial, x_t, y_t, v_x_t, v_y_t

if __name__ == '__main__':
    queries = Queries()
    queries.check_velocity_range(0.5, [[-10,10],[-10,10]], 100)








