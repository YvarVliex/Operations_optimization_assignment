# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:43:23 2021

@author: Nils de Krom
"""


import gurobipy as gb
from numpy.core.fromnumeric import _take_dispatcher
import pandas as pd
import numpy as np
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
        
        self.df_distance_2 = self.df_distance.reindex(self.df_deliv.columns[:-1], columns=self.df_deliv.columns[:-1]).copy()
        
        self.df_coordinates = self.txt_file_reader(self.txt_file, 0).reindex(self.df_deliv.columns[:-1])

        #Initialise some model param
        self.AZmodel = gb.Model("Azores")
        self.x_var = {}
        self.t_var = {}
        # self.D_var = {}
        # self.P_var = {}
        
        self.min_landingdist = min_landingdist
        
    # Function that allows to extract data from an excel file    
    def excel_data_obtainer(self, filename, sheetname, start_row, end_row, cols):
        df_temp = pd.read_excel(filename, sheetname, usecols = cols, skiprows = start_row, nrows= (end_row-start_row) )
        return df_temp
    
    # Function that allows to extract data from a text file
    def txt_file_reader(self, txt_file, col_indx):
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
                        temp_df.iloc[i,j] = round(self.df_distance_2.iloc[i,j]/(0.7*self.k_Vcr_dct[k])*60 + self.df_fleet["Turnaround Time (mins)"][k],3) 
                    else:
                        temp_df.iloc[i,j] = 0  
                    
            self.time_df_dct[k] = temp_df.copy()

    def initialise_model(self):
    
        # create dictionaries for x (flights between i and j), P (pickups) and D (deliveries)
        
        self.x_name = {}
        # self.P_name = {}
        # self.D_name = {}
    
        # loop through islands for starting node i and arriving node j
        for i in self.n_islands:
            for j in self.n_islands:
                if i!=j:
                    for k in range(self.num_veh):
                         # Create variables D and P to be integers for deliveries and pickups
                        # self.D_var[i,j,k] = self.AZmodel.addVar(name = f"D({i,j,k})", vtype = gb.GRB.INTEGER, lb = 0)
                        # self.P_var[i,j,k] = self.AZmodel.addVar(name = f"P({i,j,k})", vtype = gb.GRB.INTEGER, lb = 0)
                        # self.P_name[f"P({i,j,k})"] = (i,j,k)
                        # self.D_name[f"D({i,j,k})"] = (i,j,k)
                        
                        # loop through aircraft type and aircraft number and perform calculation on cost:
                        # = fuel cost * distance + 2 * landing/take-off cost (is same at all airports) + nr. seats
                        # per type of aircraft * cost per passenger
                            
                        #Maybe hier toch cost per L/km/pax doen?
                        # temp_obj_2 = self.df_fleet["Fuel cost [L/km/pax]"].iloc[k]*self.df_distance_2.iloc[i,j]*self.df_fleet["Number of Seats"][k] +\
                        #     2*self.df_cost["Landing/TO cost [Euro]"][k] + \
                        #         self.df_fleet["Number of Seats"][k]*self.df_cost["Cost per passenger"][k]
                        
                        # temp_obj = self.df_fleet["Fuel cost L/km"].iloc[k]*self.df_distance_2.iloc[i,j] +\
                        #     2*self.df_cost["Landing/TO cost [Euro]"][k] + \
                        #         self.df_fleet["Number of Seats"][k]*self.df_cost["Cost per passenger"][k]
                        self.x_var[i,j,k] = self.AZmodel.addVar(name = f"x({i,j,k})", vtype = gb.GRB.INTEGER, lb = 0)#, obj = temp_obj_2)
                        self.x_name[f"x({i,j,k})"] = (i,j,k)
                        
        for i in self.n_islands:
            for k in range(self.num_veh):
                self.t_var[i,k] = self.AZmodel.addVar(name = f"t({i,k})", vtype=gb.GRB.CONTINUOUS)
                            
        # self.AZmodel.update()

        # Set objective function, minimize this cost
        # self.AZmodel.setObjective(self.AZmodel.getObjective(), gb.GRB.MINIMIZE) 
        self.AZmodel.setObjective(gb.quicksum(self.temp_objective(i,j,k)*self.x_var[i,j,k] for i in self.n_islands for j in self.n_islands for k in range(self.num_veh) if i!=j ))
        
        # # objective 
        # model.setObjective(gb.quicksum(distances[i,j]*x[i,j,k] for i,j,k in arc_var),gb.GRB.MINIMIZE)
        
        self.AZmodel.update()
        
        
    def temp_objective(self,i,j,k):
       return self.df_fleet["Fuel cost [L/km/pax]"].iloc[k]*self.df_distance_2.iloc[i,j]*self.df_fleet["Number of Seats"][k] +\
                            2*self.df_cost["Landing/TO cost [Euro]"][k] + \
                                self.df_fleet["Number of Seats"][k]*self.df_cost["Cost per passenger"][k]
    
    def add_constraints(self):
        #when this func is called all constraints will be added
        self.practical_constr()
        self.deliv_constr()
        # self.subtour_elim_constr()
        self.time_constr()
        
        self.AZmodel.update()
    
    
    def practical_constr(self):
        
        # arrival and departures from depot
        # for k in range(self.num_veh):
        #     self.AZmodel.addConstr(gb.quicksum(self.x_var[0,j,k] for j in self.n_islands if j!=0), gb.GRB.LESS_EQUAL, 1)
        #     self.AZmodel.addConstr(gb.quicksum(self.x_var[i,0,k] for i in self.n_islands if i!=0), gb.GRB.LESS_EQUAL, 1)
        # model.addConstrs(gb.quicksum(x[0,j,k] for j in clients) <= 1 for k in vehicles)
        # model.addConstrs(gb.quicksum(x[i,0,k] for i in clients) <= 1 for k in vehicles)
        
        self.AZmodel.addConstrs(gb.quicksum(self.x_var[0,j,k] for j in self.n_islands[1:]) <= 1 for k in range(self.num_veh))
        self.AZmodel.addConstrs(gb.quicksum(self.x_var[i,0,k] for i in self.n_islands[1:]) <= 1 for k in range(self.num_veh))
        
        # sum Xij >= 1 arrive at node
        for i in self.n_islands:
            self.AZmodel.addConstr(gb.quicksum(self.x_var[i,j,k] for j in self.n_islands if j!=i for k in range(self.num_veh)), gb.GRB.EQUAL, 1, name = "Eachnodevistatleastonce")
            
        # # more than one vehicle per node
        # model.addConstrs(gb.quicksum(x[i,j,k] for j in nodes for k in vehicles if i!=j) ==1 for i in clients)
        self.AZmodel.addConstrs(gb.quicksum(self.x_var[i,j,k] for j in self.n_islands for k in range(self.num_veh) if i!=j) ==1 for i in self.n_islands[1:])
        
        # aircraft that arrives must also leave (constraint 2.5)
        # flow conservation
        for k in range(self.num_veh):
            for h in self.n_islands:
                self.AZmodel.addConstr(gb.quicksum(self.x_var[i,h,k] for i in self.n_islands if i!=h), gb.GRB.EQUAL, gb.quicksum(self.x_var[h,j,k] for j in self.n_islands if j!=h), name = "ArrivedalsoLeave")
        # flow conservation
        # model.addConstrs(gb.quicksum(x[i,j,k] for j in nodes if i!=j)-gb.quicksum(x[j,i,k] for j in nodes if i!=j)==0 for i in nodes for k in vehicles)
        
        # Q400 cannot land a corvo (eq 2.6) 
        #Add that it cannot TO
        # constr_t = 1
        node_corvo = 1
        
        for k in range(self.num_veh):
            if self.df_fleet["Landing Distance (@MLW)"][k] >= self.min_landingdist:
                self.AZmodel.addConstr(gb.quicksum(self.x_var[i,node_corvo,k] for i in self.n_islands if i!=node_corvo), gb.GRB.EQUAL, 0)
                self.AZmodel.addConstr(gb.quicksum(self.x_var[node_corvo,j,k] for j in self.n_islands if j!=node_corvo), gb.GRB.EQUAL, 0)
        
        # #Evrything must go to 0?
        # for t in range(len(self.t_dct)):
        #     for k in range(self.t_dct[t]): 
        #         self.AZmodel.addConstr(gb.quicksum(self.x_var[i,0,t,k] for i in self.n_islands), gb.GRB.GREATER_EQUAL, 1)
        
                
        
        self.AZmodel.update()
    
    def deliv_constr(self):
        # flow conservation
        
        # for k in range(self.num_veh):
        #     self.AZmodel.addConstr(gb.quicksum(self.df_deliv.iloc[0,i]*gb.quicksum(self.x_var[i,j,k] for j in self.n_islands if i!= j) for i in self.n_islands) <= self.df_fleet["Number of Seats"][k])
        
        self.AZmodel.addConstrs(gb.quicksum(self.df_deliv.iloc[0,i]*gb.quicksum(self.x_var[i,j,k] for j in self.n_islands if i!=j) for i in self.n_islands) <= self.df_fleet["Number of Seats"][k] for k in range(self.num_veh))
        # model.addConstrs(gb.quicksum(q[i]*gb.quicksum(x[i,j,k] for j in nodes if i!= j) for i in nodes) <= Q[k] for k in vehicles)
       
        self.AZmodel.update()
        
            
    def time_constr(self):       

        # for k in range(self.num_veh):        
        #     self.AZmodel.addConstr(gb.quicksum(self.x_var[i,j,k]*self.time_df_dct[k].iloc[i,j] for i in self.n_islands for j in self.n_islands if i!=j), gb.GRB.LESS_EQUAL, 7*24*60)
        
        # flow of time
        # for k in range(self.num_veh):
        #     self.AZmodel.addConstr(self.t_var[0,k], gb.GRB.EQUAL, 0)
        
        self.AZmodel.addConstrs(self.t_var[0,k] == 0 for k in range(self.num_veh)  )
        
        self.AZmodel.addConstrs((self.x_var[i,j,k] == 1 ) >>  (self.t_var[i,k]+self.time_df_dct[k].iloc[i,j] == self.t_var[j,k]) for i in self.n_islands[1:] for j in self.n_islands[1:] for k in range(self.num_veh) if i!=j)
        # model.addConstrs(t[0,k] == 0 for k in vehicles  )
        # model.addConstrs((x[i,j,k] == 1 ) >>  (t[i,k]+s[i] + times[i,j] == t[j,k]) for i in clients for j in clients for k in vehicles if i!=j)

        
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
            # self.D_links = []
            for variable in self.AZmodel.getVars():
                if variable.x > 0.99:
                    print(str(variable.VarName) + "=" + str(variable.x))
                # if "x" in variable.varName and variable.getAttr("x")>= 0.99:
                    node_i, node_j, ac_k = self.x_name[variable.varName]
                    self.links.append(((node_i,node_j), ac_k, variable.getAttr("x"))) #nodes that the link connect, which ac if flying, value of x how often it is flying
                    
                # if "D" in variable.varName:
                #     node_i, node_j = self.D_name[variable.varName]
                #     self.D_links.append(((node_i,node_j), variable.getAttr("x")))
                    
                
        
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
    
    
    def graph_sol(self):
        # obtain graphical solution
        routes = []
        trucks = []
        K = [k for k in range(self.num_veh)]
        N = self.n_islands
        for k in range(self.num_veh):
            for i in self.n_islands:
                if i!=0 and self.x_var[0,i,k].x > 0.99:
                    aux=[0,i]
                    while i!=0:
                        j=i
                        for h in self.n_islands:
                            if j!=h and self.x_var[j,h,k].x>0.9:
                                aux.append(h)
                                i=h
                    routes.append(aux)
                    trucks.append(k)
        print(routes)
        print(trucks)
        
        # calculate times
        time_accum = list()
        for n in range(len(routes)):
            for k in range(len(routes[n])-1):
                if k==0:
                    aux=[0]
                else:
                    i = routes[n][k]
                    j = routes[n][k+1]
                    t = self.time_df_dct[k].iloc[i,j]+aux[-1]
                    aux.append(t)
            time_accum.append(aux)
        
        # plotje
        # from Colour import Color
        
        colorchoice = [ 'red', 'green', 'black', 'grey']
        
        x = self.df_coordinates["x"]
        y = self.df_coordinates["y"]
        
        plt.figure(figsize=(12,5))
        plt.scatter(x,y,color='blue')
        
        plt.scatter(x[0],y[0],color='red', marker='D')
        plt.annotate("Depot",(x[0]-1,y[0]-5.5))
        
        # for i in clients:
        #     plt.annotate('$q_{%d}=%d$|$t_{%d}$=(%d$,%d$)' %(i,q[i],i,e[i],l[i]),(X[i]+1,Y[i]))
        
        # print routes
        for r in range(len(routes)):
            for n in range(len(routes[r])-1):
                i = routes[r][n]
                j = routes[r][n+1]
                plt.plot([x[i],x[j]],[y[i],y[j]], color =colorchoice[r])
        
        for r in range(len(time_accum)):
            for n in range(len(time_accum[r])):
                i = routes[r][n]
                plt.annotate('$q_{%d}=%d$ | $t_{%d}=%d$'%(i,self.df_deliv.iloc[0,i],i,time_accum[r][n]),(x[i]+1,y[i]))
            
        
        patch = [mpatches.Patch(color=colorchoice[n],label="vehcile "+str(trucks[n])+"|cap="+str(self.df_fleet["Number of Seats"][trucks[n]])) for n in range(len(trucks))]
        plt.legend(handles=patch, loc='best')
        
        
        plt.xlabel("Distance X")
        plt.ylabel("Distance Y")
        plt.title("VRP Solution")
        plt.show()
            


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
    # print(azor_v2.links)
    # print(azor_v2.n_name)
    # azor_v2.plot_start_map()
    # azor_v2.plot_end_map()
    # print(f"Status = {azor_v2.status}")
    # print(f"Objective value = {azor_v2.objectval}")
    end_t = time.time()
    
    print(f"Runtime = {end_t-start_t}")
    
