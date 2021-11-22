"""" Test document om wat dingen uit te proberen met gurobi"""

import gurobipy as gp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Model:
    def __init__(self):
        self.aircraft_types = [1, 2] # 1 = Q200, 2 = Q400
        self.airports = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.distances = [0 , 515, 508, 277, 256, 238, 252, 164 98]
        self.demand_from_hub = [9308, 32, 246, 982, 925, 560, 284, 3731, 381]
        self.demand_to_hub = [9308, 32, 246, 982, 925, 560, 284, 3731, 381]
        # order of parameters is fuel cost [L/km], landing fees, TO fees, cost all passengers, cruise speed [m/s], TT [min], capacity, amount
        self.ac_params = {1: [0.89, 51.95, 51.95, 336.996, 149, 20, 37, 2],
                          2: [1.27, 92.89, 92.89, 728.64,185, 20, 80, 4]}
        self.ac_params = [[0.89, 1.27], [51.95, 92.89], [51.95, 92.89], [336.996, 728.64], [149, 185], [20, 20], [37, 80]
                          , [2,4]]

        #still to be figured out how to do it with amount
        self.ac_amount = [2,4]

        #model parameters
        self.model = gp.Model('Model')
        self.obj_x = 0 # x variable of objective function
        self.obj_func = 0 # objective function
        self.fuel_cost = 0
        self.landing_cost = 0
        self.TO_cost = 0
        self.passenger_cost = 0

    def decision_variables(self):
        """"Needs to be fixed!!!!"""
        amount = [2, 4] # dit moet wel echt gefixt worden
        for t in self.aircraft_types:
            for k in self.ac_amount:
                for i in self.airports:
                    for j in self.airports:
                        self.x[t,k,i,j] = self.model.addVar(vtype = GRB.integer, lb=0, name = 'x_{i,j}') # this already solves the integer and >0 constraints
                        self.obj_x += self.x[t, k, i, j]
        self.model.update()

    def cost_coeff(self):
        pass

    def objective_function(self):
        pass

    def practical_constraints(self):
        pass

    def demand_constraints(self):
        pass

    def subtour_elimination(self):
        for t in range(len(self.aircraft_types)):
            for k in self.ac_amount:
                for h in range(0, len(self.airports)):
                    self.model.addConstr(gp.quicksum(self.x[t, k, i, h] for i in len(self.airports)[h])
                                  - gp.quicksum(self.x[t, k, h, j] for j in len(self.airports)[h]) == 0)





