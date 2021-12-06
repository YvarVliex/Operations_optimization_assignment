# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:21:52 2021

@author: Nils de Krom
"""

import gurobipy as gb
import numpy as np
from numpy.matrixlib.defmatrix import asmatrix
import pandas as pd
import matplotlib.pyplot as plt

def excel_data_obtainer(filename, sheetname, start_row, end_row, cols):
        df_temp = pd.read_excel(filename, sheetname, usecols = cols, skiprows = start_row, nrows= (end_row-start_row) )
        return df_temp
    
    # Function that allows to extract data from a text file
def txt_file_reader(txt_file, col_indx):
    return pd.read_csv(txt_file, index_col = col_indx)

def temp_objective(i,j,k):
       return df_fleet["Fuel cost [L/km/pax]"].iloc[k]*df_distance_2.iloc[i,j]*df_fleet["Number of Seats"][k] +\
                            2*df_cost["Landing/TO cost [Euro]"][k] + \
                                df_fleet["Number of Seats"][k]*df_cost["Cost per passenger"][k]
    
filename = "Azores_Flight_Data_v4.xlsx"
txt_file = "coordinates_airports.txt"
sheet_names = pd.ExcelFile(filename).sheet_names
df_distance = excel_data_obtainer(filename, "Distance Table", 0,9, "A:J").set_index("Islands")
df_fleet = excel_data_obtainer(filename, "AC_data", 0, 6, "A:M").set_index("Aircraft type")
df_cost = excel_data_obtainer(filename, "Cost_sheet", 0, 6, "A,E:H").set_index("Aircraft type")
        
df_deliv = excel_data_obtainer(filename, "Demand Table", 0, 2, "B,D:M").drop(0).set_index("Start").astype('float64').round(0)
# self.df_deliv = self.df_deliv.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
df_deliv.iloc[0,0] = 0
        
df_pickup = excel_data_obtainer(filename, "Demand Table", 8, 19, "B,D").set_index("End").round(0).T
# self.df_pickup = self.df_pickup.reindex(self.df_deliv.columns[:-1]).fillna(0).copy()
df_pickup.iloc[0,0] = 0

df_distance_2 = df_distance.reindex(df_deliv.columns[:-1], columns=df_deliv.columns[:-1]).copy()

df_coordinates = txt_file_reader(txt_file, 0).reindex(df_deliv.columns[:-1])




# nodes
n = len(df_distance_2)
clients = [ i for i in range(n) if i != 0]
nodes = [0] + clients
arcs = [(i,j) for i in nodes for j in nodes if i!=j]

time_df_dct = {}

k_Vcr_dct = {}

P = 47
trips = [n for n in range(P)]
# demand
# np.random.seed(1)
q = {n: df_deliv.iloc[0,n] for n in nodes}
q[0] = 0


# time windows
e = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0} # min times
l = {0:2000, 1:2000, 2:2000, 3:2000, 4:2000, 5:2000, 6:2000, 7:2000, 8:2000, 9:2000, 10:2000} # max times

e = {n:0 for n in range(11)}
l = {n:2000 for n in range(11)}

# s = {n:np.random.randint(3,5) for n in clients} # service time at node i
# s[0] = 0

# vehicles
vehicles = [k for k in range(len(df_fleet))]#[1,2,3, 4]#, 5, 6]

Q = {k:df_fleet["Number of Seats"][k] for k in range(len(df_fleet))}#{1:50, 2:50, 3:50, 4:25}#, 5:25, 6:25}


for i in vehicles:            
    k_Vcr_dct[i] = df_fleet["Speed [km/h]"][i]
    
# t_Vcr_dct
for k in vehicles:
    temp_df = pd.DataFrame(index = df_deliv.columns[:-1], columns = df_deliv.columns[:-1])
    for i in nodes:
        for j in nodes:
            if i != j:
                temp_df.iloc[i,j] = round(df_distance_2.iloc[i,j]/(0.7*k_Vcr_dct[k])*60 + df_fleet["Turnaround Time (mins)"][k],3) 
            else:
                temp_df.iloc[i,j] = 0  
            
    time_df_dct[k] = temp_df.copy()

# coordinates
X = df_coordinates["x"]#np.random.rand(len(nodes))*100
Y = df_coordinates["y"]#np.random.rand(len(nodes))*100

# time and distances
distances = {(i,j): df_distance_2.iloc[i,j] for i in nodes for j in nodes if i!=j}
times     = {(i,j,k): time_df_dct[k].iloc[i,j] for k in vehicles for i in nodes for j in nodes if i!=j}


# plotje
plt.figure(figsize=(12,5))
plt.scatter(X,Y,color='blue')

plt.scatter(X[0],Y[0],color='red', marker='D')
plt.annotate("Depot|$t_{%d}$=(%d$,%d$)" %(0,e[0],l[0]),(X[0]-1,Y[0]-5.5))

for i in clients:
    plt.annotate('$q_{%d}=%d$|$t_{%d}$=(%d$,%d$)' %(i,q[i],i,e[i],l[i]),(X[i]-0.1,Y[i]))

plt.xlabel("Distance X")
plt.ylabel("Distance Y")
plt.title("VRP Solution")
plt.show()

# Solving the problem
# arcs for the model
arc_var = [(i,j,k,p) for i in nodes for j in nodes for k in vehicles if i!=j for p in trips]
arc_times = [(i,k,p) for i in nodes for k in vehicles for p in trips]


# model
model = gb.Model("VRPTW")

# decistion variables
x = model.addVars(arc_var, vtype=gb.GRB.INTEGER, name = 'x')
t = model.addVars(arc_times, vtype=gb.GRB.CONTINUOUS, name = 't')

# objective 
model.setObjective(gb.quicksum((df_fleet["Fuel cost [L/km/pax]"].iloc[k]*df_distance_2.iloc[i,j]*df_fleet["Number of Seats"][k] +\
                            2*df_cost["Landing/TO cost [Euro]"][k] + \
                                df_fleet["Number of Seats"][k]*df_cost["Cost per passenger"][k])*x[i,j,k,p] for i,j,k,p in arc_var),gb.GRB.MINIMIZE)

# constraints
# arrival and departures from depot
model.addConstrs(gb.quicksum(x[0,j,k,p] for j in clients) <= 1 for k in vehicles for p in trips)
model.addConstrs(gb.quicksum(x[i,0,k,p] for i in clients) <= 1 for k in vehicles for p in trips)

# more than one vehicle per node
### CHECK THIS ONE
model.addConstrs(gb.quicksum(x[i,j,k,p] for j in nodes for k in vehicles for p in trips if i!=j) >=1 for i in clients)

# flow conservation
model.addConstrs(gb.quicksum(x[i,j,k,p] for j in nodes if i!=j)-gb.quicksum(x[j,i,k,p] for j in nodes if i!=j)==0 for i in nodes for k in vehicles for p in trips)


###LOOK AT THIS ONE
model.addConstrs(gb.quicksum(q[i]*gb.quicksum(x[i,j,k,p] for j in nodes if i!= j) for i in nodes) <= Q[k] for k in vehicles for p in trips)
# model.addConstrs(gb.quicksum(q[i]*gb.quicksum(x[i,j,k,p] for j in nodes if i!= j) for i in nodes) <= Q[k]*gb.quicksum(x[i,j,k,p] for j in nodes if i!= j) for k in vehicles for p in trips)

# flow of time
model.addConstrs(t[0,k,p] == 0 for k in vehicles for p in trips)
model.addConstrs((x[i,j,k,p] == 1 ) >>  (t[i,k,p]+ times[i,j,k] == t[j,k,p]) for i in clients for j in clients for k in vehicles for p in trips if i!=j)

# model.addConstrs(t[i,k] >= e[i] for i,k in arc_times)
# model.addConstrs(t[i,k] <= l[i] for i,k in arc_times)

model.Params.timeLimit = 60
# model.Params.MIPGap = 0.1


# Multiroute constraints!!!!
# model.addConstrs(t[i,k,p] <= t[i,k,p+1] for i in clients for k in vehicles for p in trips[:-1])

model.optimize()

print("Objective Function: ",str(round(model.ObjVal,2)))
for v in model.getVars():
    if v.x > 0.99:
        print(str(v.VarName) + "=" + str(v.x))

# obtain graphical solution
routes = []
trucks = []
K = vehicles
N = nodes
for k in vehicles:
    for p in trips:
        for i in nodes:
            if i!=0 and x[0,i,k,p].x > 0.99:
                aux=[0,i]
                while i!=0:
                    j=i
                    for h in nodes:
                        if j!=h and x[j,h,k,p].x>0.9:
                            aux.append(h)
                            i=h
                routes.append(aux)
                trucks.append(k)
print(routes)
print(trucks)

# # calculate times
time_accum = list()
for n in range(len(routes)):
    for k in range(len(routes[n])-1):
        if k==0:
            aux=[0]
        else:
            i = routes[n][k]
            j = routes[n][k+1]
            t = times[i,j,k]+aux[-1]
            aux.append(t)
    time_accum.append(aux)

# plotje
# from Colour import Color
import matplotlib.patches as mpatches

colorchoice = [ 'red', 'green', 'black', 'grey']

plt.figure(figsize=(12,5))
plt.scatter(X,Y,color='blue')

plt.scatter(X[0],Y[0],color='red', marker='D')
plt.annotate("Depot",(X[0]-1,Y[0]-5.5))

# for i in clients:
#     plt.annotate('$q_{%d}=%d$|$t_{%d}$=(%d$,%d$)' %(i,q[i],i,e[i],l[i]),(X[i]+1,Y[i]))

# print routes
for r in range(len(routes)):
    for n in range(len(routes[r])-1):
        i = routes[r][n]
        j = routes[r][n+1]
        plt.plot([X[i],X[j]],[Y[i],Y[j]], color =colorchoice[r])

for r in range(len(time_accum)):
    for n in range(len(time_accum[r])):
        i = routes[r][n]
        plt.annotate('$q_{%d}=%d$ | $t_{%d}=%d$'%(i,q[i],i,time_accum[r][n]),(X[i]-0.1,Y[i]))
    

patch = [mpatches.Patch(color=colorchoice[n],label="vehcile "+str(trucks[n])+"|cap="+str(Q[trucks[n]])) for n in range(len(trucks))]
plt.legend(handles=patch, loc='best')


plt.xlabel("Distance X")
plt.ylabel("Distance Y")
plt.title("VRP Solution")
plt.show()
