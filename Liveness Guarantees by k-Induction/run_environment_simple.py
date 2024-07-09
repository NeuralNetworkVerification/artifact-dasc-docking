'''
Space motion dynamics simulation. 

@auth Ieva, Udayan, Andrew, Guy, Umberto.
@created 2023
@last changed 2024
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime

initial_point = [0.6491005037804397, 0.51, -0.0013499836749265625, 0.004394075901083542]
state = np.array(initial_point, dtype=np.float32)
x_points = [initial_point[0]]
y_points = [initial_point[1]]
v_x_points = [initial_point[2]]
v_y_points = [initial_point[3]]

DOCK_RAD = 0.5  # Successful docking distance
m = 12
n = 0.001027

MAX_STEPS = 20000
MIN_STEPS = 1
TIMESTEP = 1

dockStatus = None
networkName = "ckpt_200_with_pre_post_processing.onnx"
k = 0
kSet = False

current_distance = math.sqrt(state[0] ** 2 + state[1] ** 2)

stepcount = 0
print("COUNTED STEPS: ", stepcount)

for i in range(MAX_STEPS):
    
    print("Closed form state is", state)

    dist = math.sqrt(state[0] ** 2 + state[1] ** 2)

    if dist < current_distance and not kSet:
        k= i
        kSet = True

    # Distance-based termination conditions, docking or out of bounds
    if (dist <= DOCK_RAD) and i >= MIN_STEPS:
        print("DOCKING SUCCESSFUL")
        dockStatus = True
        break

    session = onnxruntime.InferenceSession(networkName, None)
    inputName = session.get_inputs()[0].name
    outputName = session.get_outputs()[0].name
    control = session.run([outputName], {inputName: state[None,:]})[0][0]

    control = [max(min(control[0],1),-1), max(min(control[1],1),-1)]

    # Next state computation
    x_0, y_0, v_x_0, v_y_0 = state[0], state[1], state[2], state[3]

    F_x = control[0]
    F_y = control[1]
    print("Forces")
    print(F_x,F_y)
    t = TIMESTEP

    x_t = (2 * v_y_0 / n + 4 * x_0 + F_x / (m * n ** 2)) + (2 * F_y / (m * n)) * t + (-F_x / (m * n ** 2) - 2 * v_y_0 / n - 3 * x_0) * np.cos(n * t) + \
        (-2 * F_y / (m * n ** 2) + v_x_0 / n) * np.sin(n * t)
    y_t = (-2 * v_x_0 / n + y_0 + 4 * F_y / (m * n ** 2)) + (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) * t + (-3 * F_y / (2 * m)) * (t ** 2) \
        + (-4 * F_y / (m * n ** 2) + 2 * v_x_0 / n) * np.cos(n * t) + (2 * F_x / (m * n ** 2) + 4 * v_y_0 / n + 6 * x_0) * np.sin(n * t)
    v_x_t = (2 * F_y / (m * n)) + (-2 * F_y / (m * n) + v_x_0) * np.cos(n * t) + (F_x / (m * n) + 2 * v_y_0 + 3 * n * x_0) * np.sin(n * t)
    v_y_t = (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) + (-3 * F_y / m) * t + (2 * F_x / (m * n) + 4 * v_y_0 + 6 * n * x_0) * np.cos(n * t) + (4 * F_y / (m * n) - 2 * v_x_0) * np.sin(n * t)

    state = np.array([x_t, y_t, v_x_t, v_y_t], dtype=np.float32)
    x_points.append(x_t)
    y_points.append(y_t)
    v_x_points.append(v_x_t)
    v_y_points.append(v_y_t)

    stepcount = stepcount + 1
    print(i)


plt.title(f"Positional Trajectory for the Initial Point \n {initial_point}", fontsize = 10)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_points, y_points, 'o-', color='r', label='closed-form', markersize=2)
plt.savefig(str(initial_point) +  'posplt.png')
plt.clf()

plt.title(f"Velocity Trajectory for the Initial Point \n {initial_point}", fontsize = 10)
plt.xlabel("v_x")
plt.ylabel("v_y")
plt.plot(v_x_points, v_y_points, 'o-', color='r', label='closed-form', markersize=2)
plt.savefig(str(initial_point) +  'velplt.png')
plt.clf()
    
# No resolution reached, therefore timeout

if dockStatus is None:
    print("FAILURE: TIMEOUT")
