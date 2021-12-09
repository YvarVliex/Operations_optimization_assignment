# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:39:30 2021

@author: Nils de Krom, Maarten Beltman, Yvar Vliex
"""


import gurobipy as gb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
                
        self.df_deliv = self.excel_data_obtainer(self.filename, "Demand Table", 0, 2, "B,D:L").drop(0).set_index("Start").astype('float64').round(0)
        # self.df_deliv = self.df_deliv.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
        self.df_deliv.iloc[0,0] = 0
                
        self.df_pickup = self.excel_data_obtainer(self.filename, "Demand Table", 8, 19, "B,D").set_index("End").round(0).T
        # self.df_pickup = self.df_pickup.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
        self.df_pickup.iloc[0,0] = 0
        
        self.df_distance_2 = self.df_distance.reindex(self.df_deliv.columns, columns=self.df_deliv.columns).copy()
        
        self.df_coordinates = self.txt_file_reader(self.txt_file, 0).reindex(self.df_deliv.columns)
        
        # coordinates
        self.X = self.df_coordinates["x"]
        self.Y = self.df_coordinates["y"]
        
        ### Make required lists
        self.n_islands = len(self.df_distance_2)
        
        self.destinations = [i for i in range(1, self.n_islands)]
        self.nodes = [n for n in range(self.n_islands)]
        
        self.island_arcs = [(i,j) for i in self.nodes for j in self.nodes if i!=j]
        
        self.q = {n: self.df_deliv.iloc[0,n] for n in self.nodes}
        self.q[0] = 0
        
        # vehicles
        self.vehicles = [k for k in range(len(self.df_fleet))]
        
        self.Q = {k:self.df_fleet["Number of Seats"][k] for k in range(len(self.df_fleet))}

        self.min_landingdist = min_landingdist
        
        # Time window
        self.t_start = 0
        self.t_end = 10080-9360
        
        self.t_min = {n:self.t_start for n in range(self.n_islands)}
        self.t_max = {n:self.t_end for n in range(self.n_islands)}
        
        #Initialise some model param       
        self.AZmodel = gb.Model("Azores")
        
    # Function that allows to extract data from an excel file    
    def excel_data_obtainer(self, filename, sheetname, start_row, end_row, cols):
        df_temp = pd.read_excel(filename, sheetname, usecols = cols, skiprows = start_row, nrows= (end_row-start_row) )
        return df_temp
    
    # Function that allows to extract data from a text file
    def txt_file_reader(self, txt_file, col_indx):
        return pd.read_csv(txt_file, index_col = col_indx)
    
    def get_all_req_val(self):
        
        self.time_df_dct = {}
        self.k_Vcr_dct = {}
        
        for i in self.vehicles:            
            self.k_Vcr_dct[i] = self.df_fleet["Speed [km/h]"][i]
            
        # t_Vcr_dct
        for k in self.vehicles:
            temp_df = pd.DataFrame(index = self.df_deliv.columns, columns = self.df_deliv.columns)
            for i in self.nodes:
                for j in self.nodes:
                    if i != j:
                        temp_df.iloc[i,j] = round(self.df_distance_2.iloc[i,j]/(0.7*self.k_Vcr_dct[k])*60 + self.df_fleet["Turnaround Time (mins)"][k],3) 
                    else:
                        temp_df.iloc[i,j] = 0  
                    
            self.time_df_dct[k] = temp_df.copy()
            
        
        # time and distances
        self.distances = {(i,j): self.df_distance_2.iloc[i,j] for i in self.nodes for j in self.nodes if i!=j}
        self.times     = {(i,j,k): self.time_df_dct[k].iloc[i,j] for k in self.vehicles for i in self.nodes for j in self.nodes if i!=j}
    
        # arcs for the model
        self.arc_var = [(i,j,k) for i in self.nodes for j in self.nodes for k in self.vehicles if i!=j]
        self.arc_times = [(i,k) for i in self.nodes for k in self.vehicles]

    
    def initialise_model(self):
        # decistion variables
        self.x_var =  self.AZmodel.addVars(self.arc_var, vtype=gb.GRB.INTEGER, name = 'x')
        self.t_var =  self.AZmodel.addVars(self.arc_times, vtype=gb.GRB.CONTINUOUS, name = 't')

        # set model objective
        self.AZmodel.setObjective(gb.quicksum((self.df_fleet["Fuel cost [L/km/pax]"].iloc[k]*self.df_distance_2.iloc[i,j]*self.df_fleet["Number of Seats"][k] +\
                            2*self.df_cost["Landing/TO cost [Euro]"][k] + \
                                self.df_fleet["Number of Seats"][k]*self.df_cost["Cost per passenger"][k])*self.x_var[i,j,k] for i,j,k in self.arc_var),gb.GRB.MINIMIZE)
            
            
    def adding_constraints(self):
        
        # arrival and departures from depot
        self.AZmodel.addConstrs(gb.quicksum(self.x_var[0,j,k] for j in self.destinations) <= 1 for k in self.vehicles)
        self.AZmodel.addConstrs(gb.quicksum(self.x_var[i,0,k] for i in self.destinations) <= 1 for k in self.vehicles)
        
        # more than one vehicle per node
        self.AZmodel.addConstrs(gb.quicksum(self.x_var[i,j,k] for j in self.nodes for k in self.vehicles if i!=j) ==1 for i in self.destinations if self.q[i]>0.1)
        
        # flow conservation
        self.AZmodel.addConstrs(gb.quicksum(self.x_var[i,j,k] for j in self.nodes if i!=j)-gb.quicksum(self.x_var[j,i,k] for j in self.nodes if i!=j)==0 for i in self.nodes for k in self.vehicles)
        
        
        self.AZmodel.addConstrs(gb.quicksum(self.q[i]*gb.quicksum(self.x_var[i,j,k] for j in self.nodes if i!= j) for i in self.nodes) <= self.Q[k] for k in self.vehicles)
        
        # flow of time
        self.AZmodel.addConstrs(self.t_var[0,k] == 0 for k in self.vehicles  )
        self.AZmodel.addConstrs((self.x_var[i,j,k] == 1 ) >>  (self.t_var[i,k] + self.times[i,j,k] == self.t_var[j,k]) for i in self.destinations for j in self.destinations for k in self.vehicles if i!=j)
        
        # Not land at corvo
        node_corvo = 1
        
        for k in self.vehicles:
            if self.df_fleet["Landing Distance (@MLW)"][k] >= self.min_landingdist:
                self.AZmodel.addConstr(gb.quicksum(self.x_var[i,node_corvo,k] for i in self.nodes if i!=node_corvo), gb.GRB.EQUAL, 0)
                self.AZmodel.addConstr(gb.quicksum(self.x_var[node_corvo,j,k] for j in self.nodes if j!=node_corvo), gb.GRB.EQUAL, 0)

        
        # Time window constraint
        self.AZmodel.addConstrs(self.t_var[i,k] >= self.t_min[i] for i,k in self.arc_times)
        self.AZmodel.addConstrs(self.t_var[i,k] <= self.t_max[i] for i,k in self.arc_times)
    
    def get_solution(self):
        self.AZmodel.optimize()
        
        print(f"Value Objective Function: {self.AZmodel.ObjVal}")
        
        
        # Print the value of the objective function (that is to be minimized) 
        # and the values of all links that are active (hence x > 0.99 because 
        # an active link has a value of 1)
        for variable in self.AZmodel.getVars():
            if variable.x > 0.99:
                print(str(variable.VarName) + "=" + str(variable.x))

        # Create list of routes and for each of the active links, add this link
        # to this list. It does so with three loops. The inner loop adds the 
        # segments (arcs/segments) to a route. Then all routes are appended to
        # one list that contains all routes together that is used for plotting
        # purposes later. The third loop does so for each vehicle in the fleet.
        self.routes = []
        self.aircrafts = []
        for k in self.vehicles:
            for i in self.nodes:
                if i!=0 and self.x_var[0,i,k].x > 0.99:
                    self.leg=[0,i]
                    while i!=0:
                        j=i
                        for h in self.nodes:
                            if j!=h and self.x_var[j,h,k].x>0.9:
                                self.leg.append(h)
                                i=h
                    self.routes.append(self.leg)
                    self.aircrafts.append(k)
        print(self.routes)
        print(self.aircrafts)

        # Calculate time arrays
        self.time_accum = list()
        for n in range(len(self.routes)):
            for k in range(len(self.routes[n])-1):
                if k==0:
                    self.time_lst=[0]
                else:
                    i = self.routes[n][k]
                    j = self.routes[n][k+1]
                    t = self.times[i,j,k]+self.time_lst[-1]
                    self.time_lst.append(t)
            self.time_accum.append(self.time_lst)
        
        
    # Make a plot of the nodes (destinations with black marker, depot with red
    # marker). Also, for each node, the demand (q) is indicated. 
    def plot_nodes_map(self):

        self.textoffset = 0.1

        plt.figure(figsize=(12,5))
        plt.scatter(self.X,self.Y,color='black')

        plt.scatter(self.X[0],self.Y[0],color='red',marker='D')
        plt.annotate("Depot",(self.X[0]-self.textoffset,
                              self.Y[0]-self.textoffset))

        for i in self.destinations:
            plt.annotate('$q_{%d}={%d}$'%(i,self.q[i]),(self.X[i]-
                         self.textoffset,self.Y[i]-self.textoffset))

        plt.xlabel('Latitude $[\deg]$')
        plt.ylabel('Longitud $[\deg]$')
        plt.title("Vehicle Routing Problem Nodes")
        plt.grid()
        plt.show()
        
    def plot_trajectories_map(self):

        self.colorchoice = ['red', 'green', 'black', 'grey', 'skyblue', 'orange']        
        self.textoffset = 0.1

        plt.figure(figsize=(12,5))
        
        plt.scatter(self.X,self.Y,color='blue')

        plt.scatter(self.X[0],self.Y[0],color='red',marker='D')
       
        for r in range(len(self.routes)):
            for n in range(len(self.routes[r])-1):
                self.i = self.routes[r][n]
                self.j = self.routes[r][n+1]

                # Add arrows on each of the routes lines to indicate direction 
                # of flight
                self.middle_x = (self.X[self.j]-self.X[self.i])*0.55+self.X[
                    self.i]
                self.middle_y = (self.Y[self.j]-self.Y[self.i])*0.55+self.Y[
                    self.i]
                self.diff_x = (self.X[self.j]-self.X[self.i])
                self.diff_y = (self.Y[self.j]-self.Y[self.i])
                plt.plot([self.X[self.i],self.X[self.j]],[self.Y[self.i],
                         self.Y[self.j]], color=self.colorchoice[r], zorder = 0)
                plt.arrow(self.middle_x, self.middle_y, self.diff_x/1000, 
                    self.diff_y/1000, shape='full',head_width=0.05,
                    color=self.colorchoice[r])

        for r in range(len(self.time_accum)):
            for n in range(len(self.time_accum[r])):
                i = self.routes[r][n]
                plt.annotate('$q_{%d}=%d$ | $t_{%d}=%d$'%(i,self.q[i],i,
                    self.time_accum[r][n]),(self.X[i]-self.textoffset,
                    self.Y[i]-self.textoffset))
                                            
            
            
        patch = [mpatches.Patch(color=self.colorchoice[n],label="vehcile "+
            str(self.aircrafts[n])+"|cap="+str(self.Q[self.aircrafts[n]])) for n 
            in range(len(self.aircrafts))]
        plt.legend(handles=patch, loc='best')

        

        plt.xlabel('Latitude $[\deg]$')
        plt.ylabel('Longitude $[\deg]$')
        plt.title("Vehicle Routing Problem Solution")
        plt.show()
    
        
if __name__ == '__main__':
    min_landingdist = 800
    start_t = time.time()
    data_sheet = "Azores_Flight_Data_v4.xlsx"
    txt_file = "coordinates_airports.txt"
    azor_model = Azores_VR(data_sheet, txt_file,min_landingdist)
    
    azor_model.get_all_req_val()
    azor_model.initialise_model()
    azor_model.adding_constraints()
    
    azor_model.get_solution()  
    end_t_woplot = time.time()
    
    print(f"Runtime = {end_t_woplot-start_t}")
    
    azor_model.plot_nodes_map()
    azor_model.plot_trajectories_map()
    
    end_t_wplot = time.time()
    
    print(f"Runtime = {end_t_wplot-start_t}")
    
    