# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:28:58 2021

@author: Nils de Krom, Yvar Vliex
"""

import gurobipy as gb
# from numpy.core.fromnumeric import take_dispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from Azores_VR_program_v7 import Azores_VR
import scipy.stats


def create_run_model(data_file, text_file, min_runway):
    """Creates a model from a given data_file and text_file, outputs the cost"""
    start_t = time.time()
    model = Azores_VR(data_file, text_file, min_runway,  data_distance_cols = "A:J", data_deliv_cols = "B,D:L")
    model.get_all_req_val()
    model.initialise_model()
    model.adding_constraints()
    model.get_solution()
    # model.plot_trajectories_map()
    end_t = time.time()
    time_diff = end_t-start_t
    objective = model.objective_value
    return model, objective, time_diff


def change_demand_set_amount(text_file, plot_route):
    """
    Compares the cost (objective function) with a varying of the demand (change it to a step of 0.1 to have more data)
    """
    percentage_of_demand = [20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 115]
    model_02, result_02, time_diff_02 = create_run_model('Sens_analysis/datafile_0.2.xlsx', text_file, 800)
    model_04, result_04, time_diff_04 = create_run_model('Sens_analysis/datafile_0.4.xlsx', text_file, 800)
    model_06, result_06, time_diff_06 = create_run_model('Sens_analysis/datafile_0.6.xlsx', text_file, 800)
    model_08, result_08, time_diff_08 = create_run_model('Sens_analysis/datafile_0.8.xlsx', text_file, 800)
    model_10, result_10, time_diff_10 = create_run_model('Sens_analysis/datafile_1.0.xlsx', text_file, 800)
    model_11, result_11, time_diff_11 = create_run_model('Sens_analysis/datafile_1.1.xlsx', text_file, 800)
    model_03, result_03, time_diff_03 = create_run_model('Sens_analysis/datafile_0.3.xlsx', text_file, 800)
    model_05, result_05, time_diff_05 = create_run_model('Sens_analysis/datafile_0.5.xlsx', text_file, 800)
    model_07, result_07, time_diff_07 = create_run_model('Sens_analysis/datafile_0.7.xlsx', text_file, 800)
    model_09, result_09, time_diff_09 = create_run_model('Sens_analysis/datafile_0.9.xlsx', text_file, 800)
    model_105, result_105, time_diff_105 = create_run_model('Sens_analysis/datafile_1.05.xlsx', text_file, 800)
    model_115, result_115, time_diff_115 = create_run_model('Sens_analysis/datafile_1.15.xlsx', text_file, 800)
    models = [model_02, model_03, model_04, model_05, model_06, model_07, model_08, model_09, model_10,
              model_105, model_11, model_115]
    results = [result_02, result_03, result_04, result_05, result_06, result_07, result_08, result_09,
               result_10, result_105, result_11, result_115]
    times = [time_diff_02, time_diff_03, time_diff_04, time_diff_05, time_diff_06, time_diff_07, time_diff_08, time_diff_09,
             time_diff_10, time_diff_105, time_diff_11, time_diff_115]
    plt.scatter(percentage_of_demand, results, marker='o')
    plt.grid()
    plt.xlabel('Percentage of original demand')
    plt.ylabel('Objective function value')
    plt.show()

    pearson_obj = scipy.stats.pearsonr(np.array(percentage_of_demand),np.array(results))
    print(f'The Pearson Correlation Coefficient for demand and objective value is {pearson_obj[0]} with a p value of {pearson_obj[1]}')

    plt.scatter(percentage_of_demand, times, marker='o')
    plt.grid()
    plt.xlabel('Percentage of original demand')
    plt.ylabel('Computational time [s]')
    plt.show()

    pearson_times = scipy.stats.pearsonr(np.array(percentage_of_demand), np.array(times))
    print(f'The Pearson Correlation Coefficient for demand and time is {pearson_times[0]} with a p value of {pearson_times[1]}')

    if plot_route:
        for model in models:
            model.plot_end_map()
            plt.show()


def upgrade_corvo(datafile, textfile, desired_runway_length):
    """" Function to see the effect of upgrading the runway at Corvo so both plane types can land"""
    start_t = time.time()
    model = Azores_VR(datafile, textfile, desired_runway_length, data_distance_cols = "A:J", data_deliv_cols = "B,D:L", landingcorvoconstr = 'no')
    model.get_all_req_val()
    model.initialise_model()
    model.adding_constraints()
    model.get_solution()
    end_t_woplot = time.time()
    print(f"Runtime = {end_t_woplot - start_t}")
    model.plot_trajectories_map()
    plt.show()


def change_single_demand(datafile, textfile, desired_runway_length, node_to, new_demand):
    """
    Change the demand of a single island (or more?), islands should be put in as indices
    """
    start_t = time.time()
    model = Azores_VR(datafile, textfile, desired_runway_length, data_distance_cols = "A:J", data_deliv_cols = "B,D:L")
    model.df_deliv.iloc[0, node_to]= new_demand
    model.df_pickup.iloc[0, node_to] = new_demand
    model.get_all_req_val()
    model.initialise_model()
    model.adding_constraints()
    model.get_solution()
    end_t_woplot = time.time()
    print(f"Runtime = {end_t_woplot - start_t}")
    model.plot_trajectories_map()
    plt.show()

def change_cost(datafile, textfile, desired_runway_length, percentages):
    model = Azores_VR(datafile, textfile, desired_runway_length, data_distance_cols = "A:J", data_deliv_cols = "B,D:L")
    # print(model.df_cost.iloc[0,0])
    row_length = 6
    column_length = 3
    for percentage in percentages:
        for i in range(row_length):
            for j in range(column_length):
                model.df_cost.iloc[i,j] = model.df_cost.iloc[i,j]*percentage
    print(model.df_cost)


def remove_islands(datafile, textfile, min_runway_length, islands_to_remove):
    """Check the route and create plot for adding or removing islands. Note that islands to remove must be a filled in
    as strings with the name of the island"""
    model = Azores_VR(datafile, textfile, min_runway_length, data_distance_cols = "A:J", data_deliv_cols = "B,D:L")
    for index in islands_to_remove:
        drop_row = model.df_distance.drop(labels=index)
        drop_column = drop_row.drop(labels=index, axis=1)
        model.df_distance = drop_column
        new_deliv = model.df_deliv.drop(labels=index, axis=1)
        model.df_deliv = new_deliv
        new_pickup = model.df_pickup.drop(labels=index, axis=1)
        model.df_pickup = new_pickup

    model.get_all_req_val()
    model.initialise_model()
    model.adding_constraints()
    model.get_solution()
    model.plot_trajectories_map()


def change_hub(datafile, textfile, min_runway_length, new_hub, data_distance_cols, data_deliv_cols):
    """Change the location of the hub"""
    start_t = time.time()
    model = Azores_VR(datafile, textfile, min_runway_length, data_distance_cols, data_deliv_cols)
    all_islands = ['São Miguel', 'Corvo', 'Flores', 'Faial', 'Pico', 'São Jorge', 'Graciosa', 'Terceira', 'Santa Maria']
    new_indices = [new_hub]
    for island in all_islands:
        if island != new_hub:
            new_indices.append(island)
    # print(new_indices)
    # rearrange the matrices to make the new hub appear in row 0 and column 0
    model.df_distance_2 = model.df_distance_2.reindex(new_indices)
    model.df_distance_2 = model.df_distance_2.reindex(new_indices, axis=1)
    # print(model.nodes)
    # print(model.df_distance_2)
    model.df_coordinates = model.df_coordinates.reindex(new_indices)
    #pick up and delivery matrices
    model.df_pickup2 = model.excel_data_obtainer("Azores_Flight_Data_v4.xlsx", "Demand Table", 8, 19, "B,D").set_index("End").round(0).T
    model.df_pickup = model.df_pickup2.reindex(new_indices, axis=1)
    model.df_pickup.iloc[0,0] = 0
    model.df_deliv2 = model.excel_data_obtainer(model.filename, "Demand Table", 0, 2, "B,D:L").drop(0).set_index(
        "Start").astype('float64').round(0)
    model.df_deliv = model.df_deliv2.reindex(new_indices, axis=1).copy()
    model.df_deliv.iloc[0,0] = 0
    model.get_all_req_val()
    model.initialise_model()
    model.adding_constraints()
    model.get_solution()
    end_t_woplot = time.time()
    model.plot_trajectories_map()
    print(f"Runtime = {end_t_woplot - start_t}")
    print(f"Objective value = {model.objective_value}")
    plt.show()

data_sheet = "Azores_Flight_Data_v4.xlsx"
txt_file = "coordinates_airports.txt"
data_distance_cols = "A:J"
data_deliv_cols = "B,D:L"
min_runway = 800


"""Change Demand"""
# vary the total demand and check the results
# change_demand_set_amount(txt_file, False)

# vary the demand of a specific island
# change_single_demand(data_sheet, txt_file, 800, 8, 31)  # reduce demand of island that has the highest demand
# change_single_demand(data_sheet, txt_file, 800, 2, 36)  # increase demand of island that has the lowest

"""Change Cost"""
change_cost(data_sheet, txt_file, min_runway, [0.1])

"""Change Nodes"""
# upgrade runway of Corvo
# upgrade_corvo(data_sheet, txt_file, 1450)

# add islands
# change_single_demand(data_sheet, txt_file, 800, 5, 30)  # adding an island is simply done by increasing the demand of São Jorge above 0
# change_single_demand(data_sheet, txt_file, 800, 5, 37)  # limit case
# remove islands
# remove_islands(data_sheet, txt_file, min_runway, ['Santa Maria', 'Terceira', 'São Jorge'])
# remove_islands(data_sheet, txt_file, min_runway, ['Flores', 'Pico', 'São Jorge'])

# change hub location
# all_islands = ['São Miguel', 'Corvo', 'Flores', 'Faial', 'Pico', 'São Jorge', 'Graciosa', 'Terceira', 'Santa Maria']
# for island in all_islands:
#     change_hub(data_sheet, txt_file, min_runway, island, data_distance_cols, data_deliv_cols)