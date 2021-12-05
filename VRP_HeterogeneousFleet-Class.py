"""
Created on Sun Dec 5 20:18:43 2021

@authors: Maarten Beltman
"""

# Import Modules
import gurobipy as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Generate class
class Azores_VR:

    # Define initial values: nr. of nodes, list of destinations and list of 
    # nodes. Also creade an arc between each of the nodes as long as i!=j. 
    # Additionally define the demand q for each node and the capacity Q of the
    # vehicle. Finally, X and Y represent latitude and longitude coordinates 
    # of the islands. From this information, the distances between each of the
    # nodes are calculated
    def __init__(self):   
        self.filename = "Azores_Flight_Data_v3.xlsx"    
        self.nr_nodes = 9
        self.destinations = [i for i in range(1,self.nr_nodes) ]
        self.nodes = [i for i in range(self.nr_nodes)]
        self.arcs = [(i,j) for i in self.nodes for j in self.nodes if i!=j]  
        
        # Obtain airport coordinates in same order as the airport indices
        self.X = [-25.710126851658337, -31.11362470019505, -31.132089634693635, 
                  -28.715014937042806, -28.44135603209925, -28.169865355343898, 
                  -28.028601283425516, -27.085298004057375, -25.17094296546243]
        self.Y = [37.7459528686296, 39.67106226859879, 39.458113165093444, 
                  38.51988992329496, 38.55430695035533, 38.66328770011398, 
                  39.09235455113123, 38.75703977960301, 36.97374722219599]
        
        # Obtain distance data and reorder it such that the depot gets 0th index
        # and the other islands also get corresponding indices. Also define the
        # aircraft fleet and capacity of respective aircraft
        self.df_distance = self.excel_data_obtainer(self.filename, 
            "Distance Table", 0,9, "A:J").set_index("Islands")

        self.df_deliv = self.excel_data_obtainer(self.filename, "Demand Table", 
            0, 2, "B,D:M").drop(0).set_index("Start").astype('float64').round(0)

        self.df_deliv.iloc[0,0] = 0
        self.df_deliv.iloc[0,5] = 40
        self.q = {i: self.df_deliv.iloc[0,i] for i in self.nodes}
        self.vehicles = [1,2,3,4]
        self.Q = {1:80, 2:80, 3:80, 4:80}

        # Sample data for now to overwrite self.q earlier on
        self.q = {0:0, 1:32, 2:6, 3:22, 4:45, 5:1, 6:44, 7:51, 8:61}

        # Define minimum and maximum time windows and turn-around times
        self.t_min = {n:0 for n in range(self.nr_nodes)}
        self.t_max = {n:604800 for n in range(self.nr_nodes)}
        self.t_airport = {n:3600 for n in range(self.nr_nodes)}

        # Correctly make sure that the depot becomes island 0 in distance matrix
        self.df_distance_2 = self.df_distance.reindex(self.df_deliv.columns[
            :-1], columns=self.df_deliv.columns[:-1]).copy()

        # Create dictionary with all distances between nodes i and j
        self.distances = {(i,j): self.df_distance_2.iloc[i,j] for i in 
                          self.nodes for j in self.nodes if i!=j}

        # For now assume time = distance (UPDATE LATER WITH THE ACTUAL VALUES 
        # THAT WE HAVE!!!!!)
        self.times = self.distances
        

    # Function that allows to extract data from an excel file using pandas  
    def excel_data_obtainer(self, filename, sheetname, start_row, end_row, cols):
        df_temp = pd.read_excel(filename, sheetname, usecols = cols, skiprows = 
            start_row, nrows= (end_row-start_row) )
        return df_temp


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
        plt.show()

    # This function runs the mathematical model
    def define_model(self):
        
        # Define data for arc_var and arc_times:
        self.arc_var = [(i,j,k) for i in self.nodes for j in self.nodes for k 
            in self.vehicles if i!=j]
        self.arc_times = [(i,k) for i in self.nodes for k in self.vehicles]

        # Initialize model
        self.model = gb.Model('Classical VRP')

        # Create variables x (0 or 1) for each arc and t (time of airfact at 
        # node i). x has to be an integer, t may be continuous.
        self.x = self.model.addVars(self.arc_var, vtype = gb.GRB.INTEGER, 
                                    name = 'x')
        self.t = self.model.addVars(self.arc_times, vtype = gb.GRB.CONTINUOUS, 
                                    name = 't')
        
        # Set objective function (in this case still minimize the distance)
        # (NOTE: THIS NEEDS TO BE CHANGED WHEN ENTERING COSTS LATER ON)
        self.model.setObjective(gb.quicksum(self.distances[i,j]*self.x[i,j,k] 
                                for i,j,k in self.arc_var),gb.GRB.MINIMIZE)
        
        # Add constraint that each vehicle must leave and depart from depot once
        self.model.addConstrs(gb.quicksum(self.x[0,j,k] for j in 
            self.destinations) == 1 for k in self.vehicles)
        self.model.addConstrs(gb.quicksum(self.x[i,0,k] for i in 
            self.destinations) == 1 for k in self.vehicles)

        # Add constraint for one vehicle per node
        self.model.addConstrs(gb.quicksum(self.x[i,j,k] for j in self.nodes 
            for k in self.vehicles if i!=j) ==1 for i in self.destinations)
    
        # Flow conservation constraint:
        self.model.addConstrs(gb.quicksum(self.x[i,j,k] for j in self.nodes if 
            i!=j)-gb.quicksum(self.x[j,i,k] for j in self.nodes if i!=j)==0 for 
            i in self.nodes for k in self.vehicles)
   

        # Add constraint that the total of passengers on the route of the plane 
        # is smaller than the capacity of the plane
        self.model.addConstrs(gb.quicksum(self.q[i]*gb.quicksum(self.x[i,j,k] 
            for j in self.nodes if i!= j) for i in self.nodes) <= self.Q[k] 
            for k in self.vehicles)

        # Add constraint for flow of time
        self.model.addConstrs(self.t[0,k] == 0 for k in self.vehicles)
        self.model.addConstrs((self.x[i,j,k] == 1 ) >>  (self.t[i,k]+
            self.t_airport[i] + self.times[i,j] == self.t[j,k]) for i in 
            self.destinations for j in self.destinations for k in self.vehicles 
            if i!=j)

        # Add constraint that the time of arrival of the plane is larger than 
        # the minimum time and smaller than the maximum time of the time window
        self.model.addConstrs(self.t[i,k] >= self.t_min[i] for i,k in 
            self.arc_times)
        self.model.addConstrs(self.t[i,k] <= self.t_max[i] for i,k in 
            self.arc_times)

        # Set optimization parameters
        # model.Params.timeLimit = 60
        # model.Params.MIPGap = 0.1

        self.model.optimize()

    # Function that processes the results from the GUROBI model. 
    def process_results(self):
        
        # Print the value of the objective function (that is to be minimized) 
        # and the values of all links that are active (hence x > 0.99 because 
        # an active link has a value of 1)
        print("Objective Function: ",str(round(self.model.ObjVal,2)))
        for v in self.model.getVars():
            if v.x > 0.99:
                print(str(v.VarName) + "=" + str(v.x))

        # Create list of routes and for each of the active links, add this link
        # to this list. It does so with three loops. The inner loop adds the 
        # segments (arcs/segments) to a route. Then all routes are appended to
        # one list that contains all routes together that is used for plotting
        # purposes later. The third loop does so for each vehicle in the fleet.
        self.routes = []
        self.trucks = []
        self.K = self.vehicles
        self.N = self.nodes
        for k in self.vehicles:
            for i in self.nodes:
                if i!=0 and self.x[0,i,k].x > 0.99:
                    self.leg=[0,i]
                    while i!=0:
                        j=i
                        for h in self.nodes:
                            if j!=h and self.x[j,h,k].x>0.9:
                                self.leg.append(h)
                                i=h
                    self.routes.append(self.leg)
                    self.trucks.append(k)
        print(self.routes)
        print(self.trucks)

        # Calculate time arrays
        self.time_accum = list()
        for n in range(len(self.routes)):
            for k in range(len(self.routes[n])-1):
                if k==0:
                    self.time_lst=[0]
                else:
                    i = self.routes[n][k]
                    j = self.routes[n][k+1]
                    t = self.times[i,j]+self.t_airport[i]+self.time_lst[-1]
                    self.time_lst.append(t)
            self.time_accum.append(self.time_lst)

    # Make a plot of the nodes (destinations with black marker, depot with red
    # marker). Also, for each node, the demand (q) and time (t) is indicated. 
    # The routes that were obtained in def process_results() are plotted. This 
    # yields the soluiton of the classical vehicle routing problem
    def plot_routes_map(self):

        self.colorchoice = [ 'red', 'green', 'black', 'grey']        
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
                         self.Y[self.j]], color=self.colorchoice[r])
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
            str(self.trucks[n])+"|cap="+str(self.Q[self.trucks[n]])) for n 
            in range(len(self.trucks))]
        plt.legend(handles=patch, loc='best')


        plt.xlabel('Latitude $[\deg]$')
        plt.ylabel('Longitude $[\deg]$')
        plt.title("Vehicle Routing Problem Solution")
        plt.show()
        

# Runs the classes/functions
if __name__ == '__main__':
    azor = Azores_VR()
    azor.plot_nodes_map()
    azor.define_model()
    azor.process_results()
    azor.plot_routes_map()