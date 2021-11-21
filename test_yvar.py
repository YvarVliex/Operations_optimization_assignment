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
        # order of parameters is fuel cost [L/km], landing fees, TO fees, cost all passengers, cruise speed [m/s], TT [min], capacity
        self.ac_params = {1: [0.89, 51.95, 51.95, 336.996, 149, 20, 37],
                          2: [1.27, 92.89, 92.89, 728.64,185, 20, 80]}



    def objective_function(self):
        pass

    def practical_constraints(self):
        pass

    def demand_constraints(self):
        pass

    def subtour_elimination(self):
        pass

    def variable_constraints(self):
        pass



