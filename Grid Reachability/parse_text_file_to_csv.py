import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS_LIST = ["orange" ,"blue", "red", "green", "black", "yellow", "cyan", "purple", "grey", "forestgreen"]


def generate_df_from_text_file(path_to_text_file, path_to_output_csv, save_csv):
    all_rows = []
    column_strings = ["source_x_lb", "source_x_ub", "source_x_sign",
                      "source_y_lb", "source_y_ub", "source_y_sign",
                      "source_v_x_lb", "source_v_x_ub", "source_v_x_sign",
                      "source_v_y_lb", "source_v_y_ub", "source_v_y_sign",
                      "target_x_lb", "target_x_ub", "target_x_sign",
                      "target_y_lb", "target_y_ub", "target_y_sign",
                      "target_v_x_lb", "target_v_x_ub", "target_v_x_sign",
                      "target_v_y_lb", "target_v_y_ub", "target_v_y_sign"]


    text_file = open(path_to_text_file, "r")
    for line in text_file.readlines():
        line = line.strip()
        assert " -> " in line
        string_of_source, string_of_target = line.split(" -> ")
        tuple_of_source, tuple_of_target = eval(string_of_source), eval(string_of_target)

        # extract SOURCE information
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

        # extract TARGET information
        target_x_lb, target_x_ub = tuple_of_target[0]
        target_x_sign = "positive" if target_x_lb > 0 else ("negative" if target_x_ub < 0 else "both")
        target_y_lb, target_y_ub = tuple_of_target[1]
        target_y_sign = "positive" if target_y_lb > 0 else ("negative" if target_y_ub < 0 else "both")
        target_v_x_lb, target_v_x_ub = tuple_of_target[2]
        target_v_x_sign = "positive" if target_v_x_lb > 0 else ("negative" if target_v_x_ub < 0 else "both")
        target_v_y_lb, target_v_y_ub = tuple_of_target[3]
        target_v_y_sign = "positive" if target_v_y_lb > 0 else ("negative" if target_v_y_ub < 0 else "both")
        target_results_list = [target_x_lb, target_x_ub, target_x_sign,
                               target_y_lb, target_y_ub, target_y_sign,
                               target_v_x_lb, target_v_x_ub, target_v_x_sign,
                               target_v_y_lb, target_v_y_ub, target_v_y_sign]


        full_row_results = source_results_list + target_results_list
        all_rows.append(full_row_results)


    df = pd.DataFrame(all_rows, columns=column_strings)

    if save_csv:
        df.to_csv(path_or_buf=path_to_output_csv, index=False)

    return df


def analyze_results_for_targets(df):
    velocity_dict = {combination:[] for combination in (("positive", "positive"),
                                                        ("positive", "negative"),
                                                        ("negative", "positive"),
                                                        ("negative", "negative"),
                                                        ("positive", "both"),
                                                        ("negative", "both"),
                                                        ("both", "positive"),
                                                        ("both", "negative"),
                                                        ("both", "both"))}
    for index, row in df.iterrows():
        v_x_velocity_sign = row["target_v_x_sign"]
        v_y_velocity_sign = row["target_v_y_sign"]
        velocity_combination = (v_x_velocity_sign, v_y_velocity_sign)
        x_result = np.average((row["target_x_lb"], row["target_x_ub"]))
        y_result = np.average((row["target_y_lb"], row["target_y_ub"]))
        velocity_dict[velocity_combination].append((x_result, y_result))


    # plot results
    # for index, target_velocity_combination in enumerate(velocity_dict):
    #     results_per_combination = velocity_dict[target_velocity_combination]
    #     plt.scatter([tup[0] for tup in results_per_combination], [tup[1] for tup in results_per_combination], color=COLORS_LIST[index],
    #                 label='closed-form', s=10)
    #
    # # # plt.title(f"Targets when initialized  for -> {initial_point}")
    # # plt.xlabel("x")
    # # plt.ylabel("y")
    # # # plt.savefig(
    # # #     f"docking_with_relaxation/{LAYER_SIZE}_{LAYER_SIZE}_3216results/{LAYER_SIZE}_layersize_closedformlocations_from_" + "_".join(
    # # #         [str(s) for s in initial_point][:2]) + ".png")
    # #
    # # plt.show()
    # # plt.clf()


if __name__ == '__main__':
    PATH_TO_TEXT_FULE_WITH_REACHABLE_MAPPING = r"C:\Users\USER\Desktop\docking_local\reachableMappings.txt"
    OUTPUT_PATH_TO_CSV = r"C:\Users\USER\Desktop\docking_local\parsedReachableMappings.csv"
    SAVE_CSV = False

    # df = generate_df_from_text_file(path_to_text_file=PATH_TO_TEXT_FULE_WITH_REACHABLE_MAPPING,
    #                                 path_to_output_csv=OUTPUT_PATH_TO_CSV, save_csv=SAVE_CSV)
    df = pd.read_csv(OUTPUT_PATH_TO_CSV)

    analyze_results_for_targets(df=df)