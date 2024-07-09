1) Code for verifying the liveness property by k-induction.

For running code: Run k_induction.py. Choose to modify the variables:

"NUMBER_OF_STEPS" - maximum number of steps to use (corresponds to "k_max" in the paper).

Variables defining the initial state:

"x_lb" - positional lower bound on x,
"x_ub" - positional upper bound on x,
"y_lb" - positional lower bound on y,
"y_ub" - positional upper bound on y,
"v_x_lb" - velocity lower bound on x,
"v_x_ub" - velocity upper bound on x,
"v_y_lb" - velocity lower bound on y,
"v_y_ub" - velocity upper bound on y.

"networkName" - neural network file. Note: .onnx format.

"epsilon" - a positive number that is close to zero and defines the difference between the state values during the liveness property checking.

2) Code to project and visualize the docking trajectory of a space craft.

For running code: Run run_environment_simple.py. 

"initial_point" - the starting state of the spacecraft.
"DOCK_RAD" - the docking region (e.g., 0.5 would indicate a docking region for x[-0.5, 0.5] and y[-0.5, 0.5]).
"MAX_STEPS" - maximum timesteps to calculate.
"MIN_STEPS" - minimum timesteps to use in the calculations.
"TIMESTEP" - timestep size to use (in seconds).


