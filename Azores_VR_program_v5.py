# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:24:16 2021

@author: Nils de Krom
"""
import gurobipy as gb
from numpy.core.fromnumeric import _take_dispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class Azores_VR:
    
    def __init__(self, filename, txt_file, min_landingdist):
        
        #Obtaining all necessary panda dataframes (distance_2/fleet/cost/deliveries/pickups/coordinates)
        self.filename = filename
        self.txt_file = txt_file
        self.sheet_names = pd.ExcelFile(filename).sheet_names
        self.df_distance = self.excel_data_obtainer(self.filename, "Distance Table", 0,9, "A:J").set_index("Islands")
        self.df_fleet = self.excel_data_obtainer(self.filename, "AC_data", 0, 6, "A:M").set_index("Aircraft type")
        self.df_cost = self.excel_data_obtainer(self.filename, "Cost_sheet", 0, 6, "A,E:H").set_index("Aircraft type")
                
        self.df_deliv = self.excel_data_obtainer(self.filename, "Demand Table", 0, 2, "B,D:M").drop(0).set_index("Start").astype('float64').round(0)
        # self.df_deliv = self.df_deliv.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
        self.df_deliv.iloc[0,0] = 0
                
        self.df_pickup = self.excel_data_obtainer(self.filename, "Demand Table", 8, 19, "B,D").set_index("End").round(0).T
        # self.df_pickup = self.df_pickup.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
        self.df_pickup.iloc[0,0] = 0
        
        self.df_distance_2 = self.df_distance.reindex(self.df_deliv.columns[:-1], columns=self.df_deliv.columns[:-1]).copy()
        
        self.df_coordinates = self.txt_file_reader(self.txt_file, 0).reindex(self.df_deliv.columns[:-1])
        
        #Initialise some model param
        self.AZmodel = gb.Model("Azores")
        self.x_var = {}
        self.D_var = {}
        self.P_var = {}
        
        self.min_landingdist = min_landingdist
        
    # Function that allows to extract data from an excel file    
    def excel_data_obtainer(self, filename, sheetname, start_row, end_row, cols):
        df_temp = pd.read_excel(filename, sheetname, usecols = cols, skiprows = start_row, nrows= (end_row-start_row) )
        return df_temp
    
    # Function that allows to extract data from a text file
    def txt_file_reader(self, filename, col_indx):
        return pd.read_csv(txt_file, index_col = col_indx)
    
    # Function that creates dictionaries with type of aircraft and names of islands + indices
    def get_all_req_val(self):

        self.n_islands = np.arange(len(self.df_distance_2))
        
        self.n_name = {}
        
        self.num_veh = len(self.df_fleet)
        
        for j, name in enumerate(self.df_deliv):
            if name != self.df_deliv.columns[-1]:
                self.n_name[j] = name
    

        self.time_df_dct = {}

        self.k_Vcr_dct = {}
        for i in range(len(self.df_fleet)):            
            self.k_Vcr_dct[i] = self.df_fleet["Speed [km/h]"][i]
            
        # t_Vcr_dct
        for k in self.k_Vcr_dct.keys():
            temp_df = pd.DataFrame(index = self.df_deliv.columns[:-1], columns = self.df_deliv.columns[:-1])
            for i in self.n_islands:
                for j in self.n_islands:
                    if i != j:
                        temp_df.iloc[i,j] = self.df_distance_2.iloc[i,j]/(0.7*self.k_Vcr_dct[k])*60 + self.df_fleet["Turnaround Time (mins)"][k] 
                    else:
                        temp_df.iloc[i,j] = 0  
                    
            self.time_df_dct[k] = temp_df.copy()

    def initialise_model(self):
    
        # create dictionaries for x (flights between i and j), P (pickups) and D (deliveries)
        
        self.x_name = {}
        self.P_name = {}
        self.D_name = {}
    
        # loop through islands for starting node i and arriving node j
        for i in self.n_islands:
            for j in self.n_islands:
                
                # Create variables D and P to be integers for deliveries and pickups
                self.D_var[i,j] = self.AZmodel.addVar(name = f"D({i,j})", vtype = gb.GRB.INTEGER, lb = 0)
                self.P_var[i,j] = self.AZmodel.addVar(name = f"P({i,j})", vtype = gb.GRB.INTEGER, lb = 0)
                self.P_name[f"P({i,j})"] = (i,j)
                self.D_name[f"D({i,j})"] = (i,j)
                
                # loop through aircraft type and aircraft number and perform calculation on cost:
                # = fuel cost * distance + 2 * landing/take-off cost (is same at all airports) + nr. seats
                # per type of aircraft * cost per passenger
                for k in range(self.num_veh):
                        
                    #Maybe hier toch cost per L/km/pax doen?
                    temp_obj_2 = self.df_fleet["Fuel cost [L/km/pax]"].iloc[k]*self.df_distance_2.iloc[i,j]*self.df_fleet["Number of Seats"][k] +\
                        2*self.df_cost["Landing/TO cost [Euro]"][k] + \
                            self.df_fleet["Number of Seats"][k]*self.df_cost["Cost per passenger"][k]
                    
                    temp_obj = self.df_fleet["Fuel cost L/km"].iloc[k]*self.df_distance_2.iloc[i,j] +\
                        2*self.df_cost["Landing/TO cost [Euro]"][k] + \
                            self.df_fleet["Number of Seats"][k]*self.df_cost["Cost per passenger"][k]
                    self.x_var[(i,j,k)] = self.AZmodel.addVar(name = f"x({i,j,k})", vtype = gb.GRB.INTEGER, lb = 0, \
                                                       obj = temp_obj_2)
                    self.x_name[f"x({i,j,k})"] = (i,j,k)
                            
        self.AZmodel.update()

        # Set objective function, minimize this cost
        self.AZmodel.setObjective(self.AZmodel.getObjective(), gb.GRB.MINIMIZE) 
        
        self.AZmodel.update()
        

    
    def add_constraints(self):
        #when this func is called all constraints will be added
        self.practical_constr()
        self.pick_deliv_constr()
        # self.subtour_elim_constr()
        self.time_constr()
        
        self.AZmodel.update()
    
    
    def practical_constr(self):
        
        # sum Xij >= 1 arrive at node
        for j in self.n_islands:
            self.AZmodel.addConstr(gb.quicksum(self.x_var[i,j,k] for i in self.n_islands if j!=i for k in range(self.num_veh)), gb.GRB.GREATER_EQUAL, 1, name = "Eachnodevistatleastonce")
     
        
        # aircraft that arrives must also leave (constraint 2.5)
        
        for k in range(self.num_veh):
            for h in self.n_islands:
                self.AZmodel.addConstr(gb.quicksum(self.x_var[i,h,k] for i in self.n_islands), gb.GRB.EQUAL, gb.quicksum(self.x_var[h,j,k] for j in self.n_islands), name = "ArrivedalsoLeave")

        # Q400 cannot land a corvo (eq 2.6) 
        #Add that it cannot TO
        # constr_t = 1
        node_corvo = 1
        
        for k in range(self.num_veh):
            if self.df_fleet["Landing Distance (@MLW)"][k] >= self.min_landingdist:
                self.AZmodel.addConstr(gb.quicksum(self.x_var[i,node_corvo,k] for i in self.n_islands), gb.GRB.EQUAL, 0)
                self.AZmodel.addConstr(gb.quicksum(self.x_var[node_corvo,j,k] for j in self.n_islands), gb.GRB.EQUAL, 0)
        
        # #Evrything must go to 0?
        # for t in range(len(self.t_dct)):
        #     for k in range(self.t_dct[t]): 
        #         self.AZmodel.addConstr(gb.quicksum(self.x_var[i,0,t,k] for i in self.n_islands), gb.GRB.GREATER_EQUAL, 1)
        
                
        
        self.AZmodel.update()
    
    def pick_deliv_constr(self):
        
        # Constraint on the pickups: there are no pickups on the flight from the depot to any island
        # for j in self.n_islands:
        #     self.AZmodel.addLConstr(self.P_var[0,j], gb.GRB.EQUAL, 0)
        self.AZmodel.addConstr(gb.quicksum(self.P_var[0,j] for j in self.n_islands),gb.GRB.EQUAL, 0 )
        
        # Constraint on the deliveries: there are no deliveries on the flight from any island to the depot
        # for i in self.n_islands:
        #     self.AZmodel.addLConstr(self.D_var[i,0], gb.GRB.EQUAL, 0)
        self.AZmodel.addConstr(gb.quicksum(self.D_var[i,0] for i in self.n_islands),gb.GRB.EQUAL, 0 )
        
        
         #Constr to make sure D+P is not larger than the capacity flown on the trajectory
        for i in self.n_islands:
            for j in self.n_islands:
                if j!=i:
                    self.AZmodel.addConstr((self.D_var[i,j] + self.P_var[i,j]), gb.GRB.LESS_EQUAL, gb.quicksum((self.df_fleet["Number of Seats"][k]*self.x_var[i,j,k]) for k in range(self.num_veh)))
                    
      
        
            
        self.AZmodel.update()
        
    
    def subtour_elim_constr(self):

        ### ADD CONSTR
            
       
        self.AZmodel.update()
        
    def time_constr(self):       

        for k in range(self.num_veh):        
            self.AZmodel.addConstr(gb.quicksum(self.x_var[i,j,k]*self.time_df_dct[k].iloc[i,j] for i in self.n_islands for j in self.n_islands), gb.GRB.LESS_EQUAL, 7*24*60)
        
        #need to add sth to check if a/c back at node 0 at end week
        self.AZmodel.update()


    # Function that will solve the model using Gurobi solver                
    def get_solved_model(self):
        self.AZmodel.optimize()
        self.status = self.AZmodel.status
        print(self.status)
        # self.objectval = self.AZmodel.objval
        if self.status == gb.GRB.Status.OPTIMAL:
            self.objectval = self.AZmodel.objval
            # print('***** RESULTS ******')
            # print('\nObjective Function Value: \t %g' % self.objectval)
            #Write code to obtain plotting variables
            self.links = []
            self.D_links = []
            for variable in self.AZmodel.getVars():
                if "x" in variable.varName and variable.getAttr("x")>= 0.99:
                    node_i, node_j, ac_k = self.x_name[variable.varName]
                    self.links.append(((node_i,node_j), ac_k, variable.getAttr("x"))) #nodes that the link connect, which ac if flying, value of x how often it is flying
                    
                if "D" in variable.varName:
                    node_i, node_j = self.D_name[variable.varName]
                    self.D_links.append(((node_i,node_j), variable.getAttr("x")))
                    
                
        
        elif self.status != gb.GRB.Status.INF_OR_UNBD and self.status == gb.GRB.Status.INFEASIBLE:
            self.AZmodel.computeIIS()
        
        
                
        
    # Function that plots the longitude,latitude of each of the nodes (islands)
    def plot_start_map(self):
        x = self.df_coordinates["x"]
        y = self.df_coordinates["y"]
        fig, axs = plt.subplots()
        axs.scatter(x[0], y[0], c = "r", marker = "o", s=200)
        axs.scatter(x[1:], y[1:], c = "orange", marker = "s", s=100)
        
        # Get list with island names in strings and print the names with the nodes
        # Offset determines the distance between the node and the text, ideally not too large
        # islandsnames = []
        offset = 0.05
        # for name in self.n_name.values():
        #     islandsnames.append(name)
        # for i in range(len(islandsnames)):
        #     axs.text(x[i]+offset,y[i]+offset,islandsnames[i], c='black')
        
        for i, name in enumerate(self.n_name.values()):
            axs.text(x[i]+offset,y[i]+offset,name, c='black')

        axs.set_xlabel("Longitude $[\deg]$")
        axs.set_ylabel("Latitude $[\deg]$")
        axs.set_title("Map with Island nodes Azores")
        # axs.invert_yaxis()
        axs.grid()
        axs.set_xlim(-31.5, -24.5)
        axs.set_ylim(36.7, 39.9)
        plt.show()
        
    def plot_end_map(self):

        #sample links and nodes for creating function
        self.flight_route = [0,1,2,3,7,8,0]


        x = self.df_coordinates["x"]
        y = self.df_coordinates["y"]

        # Create array with long,lat values of flight path
        self.x_array_flight = []
        self.y_array_flight = []
        for i in self.flight_route:
            self.x_array_flight.append(x[i])
            self.y_array_flight.append(y[i])

        # Scatter nodes
        fig, axs = plt.subplots()
        axs.scatter(x[0], y[0], c = "r", marker = "o", s=200)
        axs.scatter(x[1:], y[1:], c = "orange", marker = "s", s=100)
        
        # Get list with island names in strings and print the names with the nodes
        # Offset determines the distance between the node and the text, ideally not too large
        # islandsnames = []
        offset = 0.05
        # for name in self.n_name.values():
        #     islandsnames.append(name)
        # for i in range(len(islandsnames)):
        #     axs.text(x[i]+offset,y[i]+offset,islandsnames[i], c='black')
        for i, name in enumerate(self.n_name.values()):
            axs.text(x[i]+offset,y[i]+offset,name, c='black')
        
        # For each part of the journey (from index i to i+1 in self.flight_route), plot the line from lat,long[i] to lat,long[i+1]
        for i in range(len(self.flight_route)-1):
            axs.plot([self.x_array_flight[i],self.x_array_flight[i+1]],
                     [self.y_array_flight[i],self.y_array_flight[i+1]],color='black')

            # Calculate middle point of the lines and directions to plot an arrow sign at this location with the correct heading
            self.middlex = (self.x_array_flight[i+1] - self.x_array_flight[i])*0.55 + self.x_array_flight[i]
            self.middley = (self.y_array_flight[i+1] - self.y_array_flight[i])*0.55 + self.y_array_flight[i]
            self.diffx = (self.y_array_flight[i+1]-self.y_array_flight[i])
            self.diffy = (self.x_array_flight[i+1]-self.x_array_flight[i])
            axs.arrow(self.middlex, self.middley, self.diffy/100, self.diffx/100, shape='full',head_width=0.05, color='black')

        axs.set_xlabel("Longitude $[\deg]$")
        axs.set_ylabel("Latitude $[\deg]$")
        axs.set_title("Map with routes between nodes Azores")
        axs.grid()
        axs.set_xlim(-31.5, -24.5)
        axs.set_ylim(36.7, 39.9)
        plt.show()



        ## Need links and nodes
        #Links to visualise arrows for direction of planes
        #Per link, need to know A/C and amount of times travelled
        #return None


if __name__ == '__main__':
    min_landingdist = 800
    start_t = time.time()
    data_sheet = "Azores_Flight_Data_v4.xlsx"
    txt_file = "coordinates_airports.txt"
    azor_v2 = Azores_VR(data_sheet, txt_file,min_landingdist)
    azor_v2.get_all_req_val()
    azor_v2.initialise_model()
    
    azor_v2.add_constraints()
    azor_v2.get_solved_model()
    # print(azor_v2.n_name)
    # azor_v2.plot_start_map()
    # azor_v2.plot_end_map()
    # print(f"Status = {azor_v2.status}")
    # print(f"Objective value = {azor_v2.objectval}")
    end_t = time.time()
    
    print(f"Runtime = {end_t-start_t}")
    
