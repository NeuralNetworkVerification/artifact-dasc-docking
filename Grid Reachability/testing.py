import os
import pandas as pd
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import onnxruntime

import random

def is_valid_cycle(cycle_list, mapping_of_id_to_cell):
    v_x_sign_set, v_y_sign_set = set(), set()
    for index in range(len(cycle_list)):
        current_cell = mapping_of_id_to_cell[cycle_list[index]]
        current_cell_v_x_sign, current_cell_v_y_sign = current_cell[8], current_cell[11]
        v_x_sign_set.add(current_cell_v_x_sign)
        v_y_sign_set.add(current_cell_v_y_sign)
    if "both" not in v_x_sign_set and len(v_x_sign_set)==1:
        return False # invalid cycle
    if "both" not in v_y_sign_set and len(v_y_sign_set)==1:
        return False # invalid cycle
    return True # valid cycle

def check_if_specific_cycle_in_same_range(cycle_list, mapping_of_id_to_cell):
    for index in range(len(cycle_list[:-1])):
        current_cell = mapping_of_id_to_cell[cycle_list[index]]
        current_cell_x_lb, current_cell_y_lb = current_cell[0], current_cell[3]
        next_cell = mapping_of_id_to_cell[cycle_list[index + 1]]
        next_cell_x_lb, next_cell_y_lb = next_cell[0], next_cell[3]
        if (current_cell_x_lb!=next_cell_x_lb) or (current_cell_y_lb!=next_cell_y_lb):
            return False
    return True

def search_for_all_cycles(df):
    graph_edges = []
    mapping_of_cell_to_id, mapping_of_id_to_cell = {}, {}
    next_id_to_map = 0
    for _, row in df.iterrows():
        source_tuple, target_tuple = tuple(row.tolist()[:12]), tuple(row.tolist()[12:])
        if source_tuple not in mapping_of_cell_to_id:
            mapping_of_cell_to_id[source_tuple] = next_id_to_map
            mapping_of_id_to_cell[next_id_to_map] = source_tuple
            next_id_to_map += 1
        if target_tuple not in mapping_of_cell_to_id:
            mapping_of_cell_to_id[target_tuple] = next_id_to_map
            mapping_of_id_to_cell[next_id_to_map] = target_tuple
            next_id_to_map += 1

        source_id = mapping_of_cell_to_id[source_tuple]
        target_id = mapping_of_cell_to_id[target_tuple]
        graph_edges.append((source_id, target_id))

    graph_dict = {cell_id:[] for cell_id in range(next_id_to_map)}
    for source_id, target_id in graph_edges:
        if target_id not in graph_dict[source_id]:
            graph_dict[source_id].append(target_id)

    graph_object = nx.DiGraph(graph_edges)
    cycles_iterator = nx.simple_cycles(graph_object)

    set_of_initial_multicell_cycle_points = set()
    if True:
        with open('100K_cycles_new.txt', 'a') as output_file:
            same_range_counter = 0
            valid_cycle_sign_wise_counter = 0
            for index, cycle in enumerate(cycles_iterator):

                same_range_for_whole_cycle = check_if_specific_cycle_in_same_range(cycle_list=cycle,mapping_of_id_to_cell=mapping_of_id_to_cell)
                same_range_counter += same_range_for_whole_cycle

                valid_cycle_sign_wise = is_valid_cycle(cycle_list=cycle, mapping_of_id_to_cell=mapping_of_id_to_cell)
                valid_cycle_sign_wise_counter += valid_cycle_sign_wise
                if valid_cycle_sign_wise:
                    print(cycle)

                #output_file.write(f'{cycle}\n')
                if len(cycle)>1 and cycle[0] not in set_of_initial_multicell_cycle_points:
                    set_of_initial_multicell_cycle_points.add(cycle[0])

                '''
                if index == 10001000:
                    break
                if index%1000000 == 0:
                    print(index)
                '''
    '''
    with open('id_to_cell_encoding.txt', 'a') as output_file:
        for index, cell  in mapping_of_id_to_cell.items():
            output_file.write(f'{index}->{cell}\n')

    with open('cell_to_id_encoding.txt', 'a') as output_file:
        for cell, index in mapping_of_cell_to_id.items():
            output_file.write(f'{cell}->{index}\n')

    with open('simplified_graph.txt', 'a') as output_file:
        for source, target in graph_edges:
            output_file.write(f'{source}->{target}\n')

    print("*"*10)
    print(f"last index -> {index}")
    print(f"the cycle starting points are -> {set_of_initial_multicell_cycle_points}")
    for starting_cell_id in set_of_initial_multicell_cycle_points:
        print(f"{mapping_of_id_to_cell[starting_cell_id]}")
    #print(f"total cases with same range -> {same_range_counter}")
    #print(f"total cases with same velocity sign -> {valid_cycle_sign_wise_counter}")
    '''
def check_transition(initial_point, edge_val):
    state = np.array(initial_point, dtype=np.float32)

    DOCK_RAD = 0.5  # Successful docking distance
    m = 12
    n = 0.001027

    TIMESTEP = 1

    dockStatus = None
    networkName = "ckpt_200_with_pre_post_processing_4safe.onnx"

    i = 0

    while True:

        print("Closed form state is", state)

        # Distance-based termination conditions, docking or out of bounds
        if (abs(state[0]) <= DOCK_RAD and abs(state[1]) <= DOCK_RAD):
            print("DOCKING SUCCESSFUL")
            dockStatus = True
            break

        session = onnxruntime.InferenceSession(networkName, None)
        inputName = session.get_inputs()[0].name
        outputName = session.get_outputs()[0].name
        control = session.run([outputName], {inputName: state[None, :]})[0][0]

        control = [max(min(control[0], 1), -1), max(min(control[1], 1), -1)]
        # control = [control[0],control[1]]

        # Next state computation
        x_0, y_0, v_x_0, v_y_0 = state[0], state[1], state[2], state[3]
        F_x = control[0]
        F_y = control[1]
        print("Forces")
        print(F_x, F_y)
        t = TIMESTEP

        x_t = (2 * v_y_0 / n + 4 * x_0 + F_x / (m * n ** 2)) + (2 * F_y / (m * n)) * t + (
                    -F_x / (m * n ** 2) - 2 * v_y_0 / n - 3 * x_0) * np.cos(n * t) + \
              (-2 * F_y / (m * n ** 2) + v_x_0 / n) * np.sin(n * t)
        y_t = (-2 * v_x_0 / n + y_0 + 4 * F_y / (m * n ** 2)) + (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) * t + (
                    -3 * F_y / (2 * m)) * (t ** 2) \
              + (-4 * F_y / (m * n ** 2) + 2 * v_x_0 / n) * np.cos(n * t) + (
                          2 * F_x / (m * n ** 2) + 4 * v_y_0 / n + 6 * x_0) * np.sin(n * t)
        v_x_t = (2 * F_y / (m * n)) + (-2 * F_y / (m * n) + v_x_0) * np.cos(n * t) + (
                    F_x / (m * n) + 2 * v_y_0 + 3 * n * x_0) * np.sin(n * t)
        v_y_t = (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) + (-3 * F_y / m) * t + (
                    2 * F_x / (m * n) + 4 * v_y_0 + 6 * n * x_0) * np.cos(n * t) + (
                            4 * F_y / (m * n) - 2 * v_x_0) * np.sin(n * t)

        state = np.array([x_t, y_t, v_x_t, v_y_t], dtype=np.float32)

        if x_t >= edge_val or y_t >= edge_val or x_t <= -edge_val or y_t <= -edge_val:
            return False
        i = i +1
        if i == 40000:
            return True


if __name__ == '__main__':
    path_to_dir_of_cluster_text_files = "adjacency_lists_finer/adjacency_lists_finer"
    num_transitions = 0
    all_rows = []
    column_strings = ["source_x_lb", "source_x_ub", "source_x_sign",
                      "source_y_lb", "source_y_ub", "source_y_sign",
                      "source_v_x_lb", "source_v_x_ub", "source_v_x_sign",
                      "source_v_y_lb", "source_v_y_ub", "source_v_y_sign",
                      "target_x_lb", "target_x_ub", "target_x_sign",
                      "target_y_lb", "target_y_ub", "target_y_sign",
                      "target_v_x_lb", "target_v_x_ub", "target_v_x_sign",
                      "target_v_y_lb", "target_v_y_ub", "target_v_y_sign"]


    for single_cell_cluster_output_local in os.listdir(path_to_dir_of_cluster_text_files):
        if not single_cell_cluster_output_local.endswith(".txt"):
            continue
        full_path_local_cluster_output = os.path.join(path_to_dir_of_cluster_text_files,
                                                      single_cell_cluster_output_local)
        text_file = open(full_path_local_cluster_output, "r")
        for line in text_file.readlines():
            line = line.strip()
            if line == "":
                continue
            print(line)
            assert "->" in line
            string_of_source, string_of_target = line.split("->")
            tuple_of_source, set_of_tuples_of_target = eval(string_of_source), eval(string_of_target)
            num_transitions += len(set_of_tuples_of_target)

            source_x_lb, source_x_ub = tuple_of_source[0]
            source_x_sign = "positive" if source_x_lb > 0 else ("negative" if source_x_ub < 0 else "both")
            source_y_lb, source_y_ub = tuple_of_source[1]
            source_y_sign = "positive" if source_y_lb > 0 else ("negative" if source_y_ub < 0 else "both")
            source_v_x_lb, source_v_x_ub = tuple_of_source[2]
            source_v_x_sign = "positive" if source_v_x_lb > 0 else ("negative" if source_v_x_ub < 0 else "both")
            source_v_y_lb, source_v_y_ub = tuple_of_source[3]
            source_v_y_sign = "positive" if source_v_y_lb > 0 else ("negative" if source_v_y_ub < 0 else "both")
            source_results_list = [source_x_lb, source_x_ub, source_x_sign,
                                   source_y_lb, source_y_ub, source_y_sign,
                                   source_v_x_lb, source_v_x_ub, source_v_x_sign,
                                   source_v_y_lb, source_v_y_ub, source_v_y_sign]

            for targe_tuple in set_of_tuples_of_target:
                # extract TARGET information
                target_x_lb, target_x_ub = targe_tuple[0]
                target_x_sign = "positive" if target_x_lb > 0 else ("negative" if target_x_ub < 0 else "both")
                target_y_lb, target_y_ub = targe_tuple[1]
                target_y_sign = "positive" if target_y_lb > 0 else ("negative" if target_y_ub < 0 else "both")
                target_v_x_lb, target_v_x_ub = targe_tuple[2]
                target_v_x_sign = "positive" if target_v_x_lb > 0 else ("negative" if target_v_x_ub < 0 else "both")
                target_v_y_lb, target_v_y_ub = targe_tuple[3]
                target_v_y_sign = "positive" if target_v_y_lb > 0 else ("negative" if target_v_y_ub < 0 else "both")
                target_results_list = [target_x_lb, target_x_ub, target_x_sign,
                                       target_y_lb, target_y_ub, target_y_sign,
                                       target_v_x_lb, target_v_x_ub, target_v_x_sign,
                                       target_v_y_lb, target_v_y_ub, target_v_y_sign]

                full_row_results = source_results_list + target_results_list
                all_rows.append(full_row_results)

    df = pd.DataFrame(all_rows, columns=column_strings)
    df.to_csv(path_or_buf="storage_refined.csv", index=False)
    print(num_transitions)
    print(df)
    print(len(df))

    '''
    df = pd.read_csv("storage.csv")
    print(df)
    print(len(df))
    search_for_all_cycles(df)
    '''

    '''
    num_found = 0
    answers = []
    i = 0
    while True:

        x = random.uniform(0.78125, 2.34375)
        y = random.uniform(0.78125, 2.34375)
        v_x = random.uniform(-0.1751953125, -0.0876953125)
        v_y = random.uniform(-0.0876953125, -0.0001953125)

        x = random.uniform(0.78125, 2.34375)
        y = random.uniform(0.78125, 2.34375)
        v_x = random.uniform(-0.1751953125, -0.0876953125)
        v_y = random.uniform(0.0001953125, 0.0876953125)

        init = [x,y,v_x,v_y]
        res = check_transition(init, 10)
        if res == False:
            answers.append(init)
            break
        i += 1

        #x,y,v_x,v_y = res
        print(i)

        if (0.78125 <= x <= 2.34375) and (0.78125 <= y <= 2.34375) and (-0.1751953125 <= v_x <= -0.0876953125) and (0.0001953125 <= v_y <= 0.0876953125):
        if (0.78125 <= x <= 2.34375) and (2.34375 <= y <= 3.90625) and (-0.0876953125 <= v_x <= -0.0001953125) and (
                0.0001953125 <= v_y <= 0.0876953125):
            answers.append(init)
            num_found += 1


    print(answers)
    '''