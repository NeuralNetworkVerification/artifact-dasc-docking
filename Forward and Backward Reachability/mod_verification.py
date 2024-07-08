"""
Adapted from:
File: Verification.py
Author(s): Tobey Shim

This file defines several higher-level functions which are useful for the overall
verification task. main() contains an example of how the functions can be combined
to perform verification in an automated manner.

To run, just execute Verification.py like any other script according to your OS/terminal.
"""

from myCell import Cell
import queue
import math
import matplotlib.pyplot as plt
import time

NUM_STEPS = 1  # Determines how many times Marabou will attempt to unroll the system by default.
MARABOU_TIMEOUT = 30  # Timeout in seconds for Marabou unrolling.
MAX_STEPS = 2 # Determines how many unroll steps are permitted before failure is reported on a cell.

# Parameter, can be tuned.
# Defines how much to expand bounds when performing inductive expansion outward.
EPSILON = 0.05
#Defines the velocity bound magnitude (-x to x)
V_BOUND = 0.14
#V_EPSILON = 0.00002

# Arguments: Cell object representing desired region to verify and bounds of current
# safe region.
# Returns: a tuple (bool, list of cell bounds)
# Bool is true if region was fully verified, false otherwise
# List contains "bad" cells on which verification failed.
def verify_region(cell_to_check, cur_safe_region=None):
    q = queue.Queue()
    # Right now we enqueue the entire cell - could consider pre-partitioning
    q.put(cell_to_check)
    badCells = []
    while not q.empty():
        curCell = q.get_nowait()
        print(curCell.bounds)
        exitCode, time_taken = curCell.unroll(cur_safe_region)
        if exitCode == "unsat":  # Verification succeeded
            print("Docking for input\n", curCell.bounds, "\nsuccessfully verified")
        else: #this is an "UNKNOWN" or "ERROR" case ... need to know how to handle
            if curCell.steps > MAX_STEPS: #if we go above a certain number of steps ... fail since we will always time out
                print("Failed to converge at", curCell.bounds)
                badCells.append(curCell)
            elif exitCode=="sat": #counterexample
                curBounds = curCell.bounds
                q.put(Cell(curBounds.copy(), curCell.steps + 1, MARABOU_TIMEOUT))
            else:
                print("Inconclusive result for input\n", curCell.bounds)
                curBounds = curCell.bounds

                # Representing the two cells coming from splitting the current cell.
                bounds1 = curBounds.copy()
                #bounds2 = curBounds.copy()
                bounds3 = curBounds.copy()
                #bounds4 = curBounds.copy()

                # Determine the width of the current cell along each of the 
                # 4 state variable "dimensions".
                x_diff = (curBounds[0][1] - curBounds[0][0]) / 2
                y_diff = (curBounds[1][1] - curBounds[1][0]) / 2
                v_x_diff = (curBounds[2][1] - curBounds[2][0]) / 2
                v_y_diff = (curBounds[3][1] - curBounds[3][0]) / 2

                # This is a simple heuristic to partition the cell dimension
                # which is currently widest across velocities and positions
                #biggest_pos = max(x_diff, y_diff)
                biggest_vel = max(v_x_diff, v_y_diff)

                #we also always choose to partition the velocity dimension which is currently widest

                # Partition the cell into 4 and add one more step.
                '''
                if biggest_pos == x_diff:
                    bounds1[0] = [curBounds[0][0], curBounds[0][0] + x_diff]
                    bounds2[0] = [curBounds[0][0] + x_diff, curBounds[0][1]]
                    bounds3[0] = [curBounds[0][0], curBounds[0][0] + x_diff]
                    bounds4[0] = [curBounds[0][0] + x_diff, curBounds[0][1]] 
                elif biggest_pos == y_diff:
                    bounds1[1] = [curBounds[1][0], curBounds[1][0] + y_diff]
                    bounds2[1] = [curBounds[1][0] + y_diff, curBounds[1][1]]
                    bounds3[1] = [curBounds[1][0], curBounds[1][0] + y_diff]
                    bounds4[1] = [curBounds[1][0] + y_diff, curBounds[1][1]]
                '''
                if biggest_vel == v_x_diff:
                    bounds1[2] = [curBounds[2][0], curBounds[2][0] + v_x_diff]
                    #bounds2[2] = [curBounds[2][0], curBounds[2][0] + v_x_diff]
                    bounds3[2] = [curBounds[2][0] + v_x_diff, curBounds[2][1]]
                    #bounds4[2] = [curBounds[2][0] + v_x_diff, curBounds[2][1]]
                elif biggest_vel == v_y_diff:
                    bounds1[3] = [curBounds[3][0], curBounds[3][0] + v_y_diff]
                    #bounds2[3] = [curBounds[3][0], curBounds[3][0] + v_y_diff]
                    bounds3[3] = [curBounds[3][0] + v_y_diff, curBounds[3][1]]
                    #bounds4[3] = [curBounds[3][0] + v_y_diff, curBounds[3][1]]
                    
                # Enqueue partition results but dependent on heuristic
                if time_taken > MARABOU_TIMEOUT:
                    q.put(Cell(bounds1, curCell.steps, MARABOU_TIMEOUT))
                    q.put(Cell(bounds3, curCell.steps, MARABOU_TIMEOUT))

    # All cells were eventually verified; 
    if len(badCells) == 0:
        print("Docking for full region", cell_to_check.bounds, "successfully verified")
    else:
        print("Holes found")
        for cell in badCells:  # Report failed cells.
            print(cell.bounds)
    return len(badCells) == 0


# dim: 0, 1, 2, 3  - denotes which dimension to expand (x, y, v_x, v_y)
# cur_safe_region is a list of lists representing bounds.
# Returns a list of bounds which expand cur_safe_region in the positive direction.
def expandPos(dim, cur_safe_region, epsilon = EPSILON):
    newCellBounds = [bounds.copy() for bounds in cur_safe_region]
    newCellBounds[dim][0] = newCellBounds[dim][1]
    newCellBounds[dim][1] += epsilon
    return newCellBounds


# dim: 0, 1 - denotes which dimension to expand (x, y, v_x, v_y)
# cur_safe_region is a list of lists representing bounds.
# Returns a list of bounds which expand cur_safe_region in the negative direction.
def expandNeg(dim, cur_safe_region, epsilon = EPSILON):
    newCellBounds = [bounds.copy() for bounds in cur_safe_region]
    newCellBounds[dim][1] = newCellBounds[dim][0]
    newCellBounds[dim][0] -= epsilon
    return newCellBounds


# Takes in the current safe region and the bounds of a successfully validated cell.
# validated_bounds must be "adjacent to" the current safe region, e.g.
# the bounds returned by expandPos or expandNeg.
# Modifies the current safe region in place to include the newly-verified area.
def augmentSafeRegion(cur_safe_region, validated_bounds):
    for i in range(4):
        cur_safe_region[i] = [min(cur_safe_region[i][0], validated_bounds[i][0]), max(cur_safe_region[i][1], validated_bounds[i][1])]


def main():
    x_val = []
    y_val = []


    k = 0.5 / math.sqrt(2) 
    # Define starting safe region here.

    cur_safe_region = [[-0.15,0.15],[-0.15,0.15],[-V_BOUND,V_BOUND],[-V_BOUND,V_BOUND]] #because we want to prove for this constant velocity

    #keep track how long it takes to prove this docks
    start_time = time.perf_counter()
    verify_region(Cell(cur_safe_region,NUM_STEPS,MARABOU_TIMEOUT))
    end_time = time.perf_counter()
    x_val.append(abs(cur_safe_region[0][0]))
    y_val.append(end_time-start_time)


    #first verify the docking region
    while not (cur_safe_region[0][0] < -0.5 and cur_safe_region[0][1] > 0.5 \
           and cur_safe_region[1][0] < -0.5 and cur_safe_region[1][1] > 0.5):
        print("At start, current safe region inside docking region is", cur_safe_region)
        #expand the safe region in the direction of the smallest absolute value of vertex for position - maintain constant velocity distance
        cur_possible = [abs(cur_safe_region[0][0]),abs(cur_safe_region[0][1]),abs(cur_safe_region[1][0]),abs(cur_safe_region[1][1])]

        pos = cur_possible.index(min(cur_possible))
        if pos == 0:
            explore_bounds = expandNeg(0,cur_safe_region)
        if pos == 1:
            explore_bounds = expandPos(0,cur_safe_region)
        if pos == 2:
            explore_bounds = expandNeg(1,cur_safe_region)
        if pos == 3:
            explore_bounds = expandPos(1,cur_safe_region)

        inductive_cell = Cell(explore_bounds, NUM_STEPS, MARABOU_TIMEOUT)
        if verify_region(inductive_cell):
            augmentSafeRegion(cur_safe_region, inductive_cell.bounds)
            if abs(cur_safe_region[0][0]) == abs(cur_safe_region[0][1]) == abs(cur_safe_region[1][0]) == abs(cur_safe_region[1][1]):
                end_time = time.perf_counter()
                x_val.append(abs(cur_safe_region[0][0]))
                y_val.append(end_time-start_time)
        else:
            break

    print(cur_safe_region)
    plt.scatter(x_val, y_val, color='b', label='closed-form', s=10)
    plt.title("Verification time for velocity range [-0.15,0.15]")
    plt.xlabel("Positional lower and upper bound from center")
    plt.ylabel("Time taken")
    plt.savefig('verificationtime.png')


if __name__ == '__main__':
    main()
