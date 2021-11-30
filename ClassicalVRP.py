"""
Created on Tue Nov 30 15:58:21 2021

@authors: Maarten Beltman
"""

# Import Modules
import gurobipy as gb
import numpy as np
import matplotlib.pyplot as plt

# Generate class
class Azores_VR:

    # Define initial values: nr. of nodes, list of destinations and list of 
    # nodes. Also creade an arc between each of the nodes as long as i!=j. 
    # Additionally define the demand q for each node and the capacity Q of the
    # vehicle. Finally, X and Y represent latitude and longitude coordinates 
    # of the islands. From this information, the distances between each of the
    # nodes are calculated (NOTE: THESE DISTANCES NEED TO BE UPDATED LATER 
    # BECAUSE WE HAVE THE ACTUAL VALUES IN KILOMETERS)
    def __init__(self):
        self.nr_nodes = 9
        self.destinations = [i for i in range(1,self.nr_nodes) ]
        self.nodes = [i for i in range(self.nr_nodes)]
        self.arcs = [(i,j) for i in self.nodes for j in self.nodes if i!=j]
        self.q = {1: 14, 2: 10, 3: 13, 4: 13, 5: 13, 6: 11, 7: 13, 8: 12, 0: 0}
        self.Q = 40
        self.X = [-25.710126851658337, -31.11362470019505, -31.132089634693635, 
                  -28.715014937042806, -28.44135603209925, -28.169865355343898, 
                  -28.028601283425516, -27.085298004057375, -25.17094296546243]
        self.Y = [37.7459528686296, 39.67106226859879, 39.458113165093444, 
                  38.51988992329496, 38.55430695035533, 38.66328770011398, 
                  39.09235455113123, 38.75703977960301, 36.97374722219599]
        self.distances = {(i,j): np.sqrt((self.X[i]-self.X[j])**2 + 
                         (self.Y[i]-self.Y[j])**2) for i in self.nodes for j 
                         in self.nodes if i!=j}
        
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

        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title("Vehicle Routing Problem Nodes")
        plt.show()

    # This function runs the mathematical model
    def define_model(self):

        # Initialize model
        self.model = gb.Model('Classical VRP')

        # Create variables x (0 or 1 for now) for each arc and u (temporary
        # capacity in the plane). If u = 0, it means the entire plane is full.
        # If u = Q, it means the entire plane is empty.
        self.x = self.model.addVars(self.arcs, vtype = gb.GRB.BINARY, 
                                    name = 'x')
        self.u = self.model.addVars(self.destinations, ub=self.Q, 
                               vtype = gb.GRB.CONTINUOUS, name = 'u')
        
        # Set objective function (in this case still minimize the distance)
        # NOTE: THIS NEEDS TO BE CHANGED WHEN ENTERING COSTS LATER ON
        self.model.setObjective(gb.quicksum(self.distances[i,j] * self.x[i,j] 
                                for i,j in self.arcs), gb.GRB.MINIMIZE)
        
        # Add constraint that each node must arrived at once and must be 
        # departed from once
        self.model.addConstrs(gb.quicksum(self.x[i,j] for j in self.nodes if 
                              i!=j) == 1 for i in self.destinations)
        self.model.addConstrs(gb.quicksum(self.x[i,j] for i in self.nodes if 
                              i!=j) == 1 for j in self.destinations)

        # Add constraint that in the case an arc has a value of 1 (hence is 
        # active), then it should follow that the capacity u after visiting
        # node j is equal to the capacity of u before visiting node j (hence
        # the capacity of u after visiting node i) + the number of deliveries
        # at node j: since q[j] are delivered at node j, the capacity increases
        # since there is more free space in the plane available
        # NOTE: WE WILL NEED TO DO SOMETHING WITH THIS WHEN ADDING PICKUPS
        self.model.addConstrs((self.x[i,j] == 1) >> (self.u[i] + self.q[j] == 
                              self.u[j]) for i,j in self.arcs if i!=0 and j!=0)
        
        # Add constraint that the capacity after delivering q[i] at node i is 
        # always higher or equal than q[i] itself such that it makes sure that
        # the space occupied previously with deliverables is now free
        self.model.addConstrs(self.u[i] >= self.q[i] for i in 
                              self.destinations)

        # Add constraint that the capacity at node i is smaller than the total
        # capacity (Q) of the plane at any destination i
        self.model.addConstrs(self.u[i] <= self.Q for i in self.destinations)

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
        # to this list. It does so with two loops. The inner loop adds the 
        # segments (arcs/segments) to a route. Then all routes are appended to
        # one list that contains all routes together that is used for plotting
        # purposes later
        self.routes = list()
        for i in self.destinations:
            if self.x[(0,i)].x > 0.99:
                self.link = [0,i]
                while i!=0:
                    j = i
                    for k in self.nodes:
                        if j!=k and self.x[(j,k)].x > 0.99:
                            self.link.append(k)
                            i = k
                self.routes.append(self.link)
        print(self.routes)

    # Make a plot of the nodes (destinations with black marker, depot with red
    # marker). Also, for each node, the demand (q) is indicated. The routes 
    # that were obtained in def process_results() are plotted. This yields the
    # soluiton of the classical vehicle routing problem
    def plot_routes_map(self):

        self.textoffset = 0.1

        plt.figure(figsize=(12,5))
        plt.scatter(self.X,self.Y,color='black')

        plt.scatter(self.X[0],self.Y[0],color='red',marker='D')
        plt.annotate("Depot",(self.X[0]-self.textoffset,
                              self.Y[0]-self.textoffset))

        for i in self.destinations:
            plt.annotate('$q_{%d}={%d}$'%(i,self.q[i]),(self.X[i]-
                         self.textoffset,self.Y[i]-self.textoffset))
        
        for r in range(len(self.routes)):
            for n in range(len(self.routes[r])-1):
                self.i = self.routes[r][n]
                self.j = self.routes[r][n+1]
                plt.plot([self.X[self.i],self.X[self.j]],[self.Y[self.i],
                         self.Y[self.j]], color='green')

        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title("Vehicle Routing Problem Solution")
        plt.show()

# Runs the classes/functions
if __name__ == '__main__':
    azor = Azores_VR()
    azor.plot_nodes_map()
    azor.define_model()
    azor.process_results()
    azor.plot_routes_map()