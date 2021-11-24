# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:41:38 2021

@author: Nils de Krom
"""

import gurobipy as gb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class Azores_VR:
    
    def __init__(self, filename, txt_file):
        
        #Obtaining all necessary panda dataframes
        self.filename = filename
        self.txt_file = txt_file
        self.sheet_names = pd.ExcelFile(filename).sheet_names
        self.df_distance = self.excel_data_obtainer(self.filename, "Distance Table", 0,9, "A:J").set_index("Islands")
        self.df_fleet = self.excel_data_obtainer(self.filename, "AC_data", 0, 2, "A:M").set_index("Aircraft type")
        self.df_cost = self.excel_data_obtainer(self.filename, "Cost_sheet", 0, 2, "A,E:H").set_index("Aircraft type")
                
        self.df_deliv = self.excel_data_obtainer(data_sheet, "Demand Table", 0, 2, "B,D:M").drop(0).set_index("Start").astype('float64').round(0)
        self.df_deliv = self.df_deliv.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
                
        self.df_pickup = self.excel_data_obtainer(data_sheet, "Demand Table", 8, 19, "B,D").set_index("End").round(0).T
        self.df_pickup = self.df_pickup.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
        
        self.df_distance_2 = self.df_distance.reindex(self.df_deliv.columns[:-1], columns=self.df_deliv.columns[:-1]).copy()
        
        self.df_coordinates = self.txt_file_reader(self.txt_file, 0).reindex(self.df_deliv.columns[:-1])
        #Initialise some model param
        self.AZmodel = gb.Model("Azores")
        self.x_var = {}
        self.D_var = {}
        self.P_var = {}
        
        
    def excel_data_obtainer(self, filename, sheetname, start_row, end_row, cols):
        df_temp = pd.read_excel(filename, sheetname, usecols = cols, skiprows = start_row, nrows= (end_row-start_row) )
        return df_temp
    
    def txt_file_reader(self, filename, col_indx):
        return pd.read_csv(txt_file, index_col = col_indx)
    
    def get_all_req_val(self):
        self.t_dct = {}
        for i in range(len(self.df_fleet)):            
            self.t_dct[i] = self.df_fleet["Number in fleet"][i]

        self.n_islands = np.arange(len(self.df_distance_2))
        
        self.n_name = {}
        
        for j, name in enumerate(self.df_deliv):
            if name != self.df_deliv.columns[-1]:
                self.n_name[j] = name
                
    def initialise_model(self):
    
        try:
            self.x_name = {}
            self.P_name = {}
            self.D_name = {}
        
            for i in self.n_islands:
                for j in self.n_islands:
                    
                    self.D_var[i,j] = self.AZmodel.addVar(name = f"D({i,j})", vtype = gb.GRB.INTEGER, lb = 0)
                    self.P_var[i,j] = self.AZmodel.addVar(name = f"P({i,j})", vtype = gb.GRB.INTEGER, lb = 0)
                    self.P_name[f"P({i,j})"] = (i,j)
                    self.P_name[f"D({i,j})"] = (i,j)
                    
                    for t in range(len(self.t_dct)):
                        for k in range(self.t_dct[t]):
                            temp_obj = self.df_fleet["Fuel cost L/km"].iloc[t]*self.df_distance_2.iloc[i,j] +\
                                2*self.df_cost["Landing/TO cost [Euro]"][t] + \
                                    self.df_fleet["Number of Seats"][t]*self.df_cost["Cost per passenger"][t]
                            self.x_var[(i,j,t,k)] = self.AZmodel.addVar(name = f"x({i,j,t,k})", vtype = gb.GRB.INTEGER, lb = 0, \
                                                               obj = temp_obj)
                            self.x_name[f"x({i,j,t,k})"] = (i,j,t,k)
                                
            self.AZmodel.update()
            self.AZmodel.setObjective(self.AZmodel.getObjective(), gb.GRB.MINIMIZE) 
        
        except:
            print("Error undefined variables: Run get_all_req_val() first.")
        
    
    def add_constraints(self):
        #when this func is called all constraints will be added
        self.practical_constr()
        self.pick_deliv_constr()
        self.subtour_elim_constr()
    
    
    def practical_constr(self):
        return None
    
    def pick_deliv_constr(self):
        return None
    
    def subtour_elim_constr(self):
        return None
                
    def get_solved_model(self):
        self.AZmodel.optimize()
        self.status = self.AZmodel.status
        
        #Write code to obtain plotting variables
        self.links = []
        for variable in self.AZmodel.get_vars():
            if "x" in variable.varName and variable.getAttr("x")>= 1:
                node_i, node_j, ac_t, _ = self.x_name[variable.varName]
                self.links.append((node_i,node_j), ac_t,variable.getAttr("x"))  #nodes that the link connect, which ac if flying, value of x how often it is flying
                
                
        

    def plot_start_map(self):
        x = self.df_coordinates["x"]
        y = self.df_coordinates["y"]
        fig, axs = plt.subplots()
        axs.scatter(x[0], y[0], c = "r", marker = "+")
        axs.scatter(x[1:], y[1:], c = "orange", marker = "s")
        axs.set_xlabel("Longitude $[\deg]$")
        axs.set_ylabel("Latitude $[\deg]$")
        axs.set_title("Map with Island nodes Azores")
        # axs.invert_yaxis()
        axs.grid()
        plt.show()
    
    def plot_end_map(self):
        ## Need links and nodes
        #Links to visualise arrows for direction of planes
        #Per link, need to know A/C and amount of times travelled
        return None


if __name__ == '__main__':
    start_t = time.time()
    data_sheet = "Azores_Flight_Data_v2.xlsx"
    txt_file = "coordinates_airports.txt"
    azor = Azores_VR(data_sheet, txt_file)
    azor.get_all_req_val()
    azor.initialise_model()
    
    
    # print(azor.n_name)
    # azor.plot_start_map()
    
    end_t = time.time()
    
    print(f"Runtime = {end_t-start_t}")
    
