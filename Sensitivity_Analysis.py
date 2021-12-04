"""
Created on Thu Dec 2 14:05 2021

@authors: Yvar Vliex
"""

import gurobipy as gb
# from numpy.core.fromnumeric import take_dispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from Azores_VR_program_v5 import Azores_VR
import scipy.stats

"""
TO DO: fix issue with not recognizing txt file when initializing model
"""


def create_run_model(data_file, text_file, min_runway):
    """Creates a model from a given data_file and text_file, outputs the cost"""
    model = Azores_VR(data_file, text_file, min_runway)
    model.get_all_req_val()
    model.initialise_model()
    model.add_constraints()
    model.get_solved_model()
    objective_val = model.objectval
    return model, objective_val


def change_demand_set_amount(text_file, plot_route):
    """
    Compares the cost (objective function) with a varying of the demand (all demand is varied by the same amount, this
    can be changed, so far I just put some random values)
    Only thing that is necessary is to create a bunch of datafiles with varying demand
    """
    percentage_of_demand = [20, 40, 60, 80, 100, 120, 140, 160, 180]
    model_02, result_02 = create_run_model('datafile_0.2', text_file, 800)
    model_04, result_04 = create_run_model('datafile_0.4', text_file, 800)
    model_06, result_06 = create_run_model('datafile_0.6', text_file, 800)
    model_08, result_08 = create_run_model('datafile_0.8', text_file, 800)
    model_10, result_10 = create_run_model('datafile_1.0', text_file, 800)
    model_12, result_12 = create_run_model('datafile_1.2', text_file, 800)
    model_14, result_14 = create_run_model('datafile_1.4', text_file, 800)
    model_16, result_16 = create_run_model('datafile_1.6', text_file, 800)
    model_18, result_18 = create_run_model('datafile_1.8', text_file, 800)

    models = [model_02, model_04, model_06, model_08, model_10, model_12, model_14, model_16, model_18]
    results = [result_02, result_04, result_06, result_08, result_10, result_12, result_14, result_16, result_18]
    plt.plot(percentage_of_demand, results)
    plt.grid()
    plt.xlabel('Percentage of original demand')
    plt.ylabel('Cost funtion')
    plt.show()

    spearman = scipy.stats.spearmanr(a=np.array(percentage_of_demand), b=np.array(results))
    print(f'The Spearman correlation is {spearman[0]} with a p value of {spearman[1]}')

    if plot_route:
        for model in models:
            model.plot_end_map()
            plt.show()


def upgrade_corvo(datafile, textfile, desired_runway_length):
    """" Function to see the effect of upgrading the runway at Corvo so both plane types can land"""
    model = Azores_VR(datafile, textfile, desired_runway_length)
    model.get_all_req_val()
    model.initialise_model()
    model.add_constraints()
    model.get_solved_model()
    objective_val = model.objectval
    model.plot_end_map()
    print(f'The new value of the objective function is {objective_val}')


def change_single_demand(datafile, textfile, desired_runway_length, node_to, new_demand):
    """
    Change the demand of a single island (or more?), islands should be put in as indices
    """
    model = Azores_VR(datafile, textfile, desired_runway_length)
    model.df_deliv.iloc[0, node_to]=new_demand
    model.df_pickup.iloc[0, node_to] = new_demand
    model.get_all_req_val()
    model.initialise_model()
    model.add_constraints()
    model.get_solved_model()
    objective_val = model.objectval
    model.plot_end_map()
    print(f'The new value of the objective function is {objective_val}')


def remove_islands(datafile, textfile, min_runway_length, islands_to_remove):
    """Check the route and create plot for adding or removing islands. Note that islands to remove must be a filled in
    as strings with the name of the island"""
    model = Azores_VR(datafile, textfile, min_runway_length)
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
    model.add_constraints()
    model.get_solved_model()
    objective_val = model.objectval
    model.plot_end_map()
    print(f'The new value of the objective function is {objective_val}')


def change_hub(datafile, textfile, min_runway_length, new_hub):
    """Change the location of the hub"""
    model = Azores_VR(datafile, textfile, min_runway_length)
    all_islands = ['São Miguel', 'Corvo', 'Flores', 'Faial', 'Pico', 'São Jorge', 'Graciosa', 'Terceira', 'Santa Maria']
    new_indices = [new_hub]
    for island in all_islands:
        if island != new_hub:
            new_indices.append(island)
    # rearrange the matrices to make the new hub appear in row 0 and column 0
    model.df_distance_2 = model.df_distance_2.reindex(new_indices)
    model.df_distance_2 = model.df_distance_2.reindex(new_indices, axis=1)
    model.df_coordinates = model.df_coordinates.reindex(new_indices)
    #pick up and delivery matrices
    model.df_pickup2 = model.excel_data_obtainer("Azores_Flight_Data_v4.xlsx", "Demand Table", 8, 19, "B,D").set_index("End").round(0).T
    model.df_pickup = model.df_pickup2.reindex(new_indices, axis=1)
    model.df_pickup.iloc[0,0] = 0
    model.df_deliv2 = model.excel_data_obtainer(model.filename, "Demand Table", 0, 2, "B,D:M").drop(0).set_index(
        "Start").astype('float64').round(0)
    model.df_deliv = model.df_deliv2.reindex(new_indices, axis=1)
    model.df_deliv.iloc[0,0] = 0
    model.get_all_req_val()
    model.initialise_model()
    model.add_constraints()
    model.get_solved_model()
    objective_val = model.objectval
    model.plot_end_map()
    print(f'The new value of the objective function is {objective_val}')

data_sheet = "Azores_Flight_Data_v4.xlsx"
txt_file = "coordinates_airports.txt"
min_runway = 800
change_hub(data_sheet, txt_file, min_runway, 'Corvo')









