import gurobipy as gb
import numpy as np
from numpy.matrixlib.defmatrix import asmatrix
import pandas as pd
import matplotlib.pyplot as plt

# nodes
n = 6
clients = [ i for i in range(n) if i != 0]
nodes = [0] + clients
arcs = [(i,j) for i in nodes for j in nodes if i!=j]

# demand
np.random.seed(0)
#q = {n:np.random.randint(25,30) for n in clients}
q = {0:0, 1:25, 2:25, 3:25, 4:25, 5:24}
b = {0:0, 1:25, 2:25, 3:25, 4:25, 5:25}



# time windows
e = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0} # min times
l = {0:2000, 1:2000, 2:2000, 3:2000, 4:2000, 5:2000, 6:2000, 7:2000, 8:2000, 9:2000, 10:2000} # max times

e = {n:0 for n in range(6)}
l = {n:200000 for n in range(6)}

s = {n:np.random.randint(3,5) for n in clients} # service time at node i
s[0] = 0

# vehicles
vehicles = [1,2,3,4]
Q = {1:5000, 2:5000, 3:2500, 4:2500}

# coordinates
X = np.random.rand(len(nodes))*100
Y = np.random.rand(len(nodes))*100

# time and distances
distances = {(i,j): np.hypot(X[i]-X[j],Y[i]-Y[j]) for i in nodes for j in nodes if i!=j}
times     = {(i,j): np.hypot(X[i]-X[j],Y[i]-Y[j]) for i in nodes for j in nodes if i!=j}

# plotje
# plt.figure(figsize=(12,5))
# plt.scatter(X,Y,color='blue')

# plt.scatter(X[0],Y[0],color='red', marker='D')
# plt.annotate("Depot|$t_{%d}$=(%d$,%d$)" %(0,e[0],l[0]),(X[0]-1,Y[0]-5.5))

# for i in clients:
#     plt.annotate('$q_{%d}=%d$|$t_{%d}$=(%d$,%d$)' %(i,q[i],i,e[i],l[i]),(X[i]+1,Y[i]))

# plt.xlabel("Distance X")
# plt.ylabel("Distance Y")
# plt.title("VRP Solution")
# plt.show()

# Solving the problem
# arcs for the model
arc_var    = [(i,j,k) for i in nodes for j in nodes for k in vehicles if i!=j]
arc_times  = [(i,j,k) for i in nodes for j in nodes for k in vehicles if i!=j]
arc_deliv  = [(i,j,k) for i in nodes for j in nodes for k in vehicles if i!=j]
arc_pickup = [(i,j,k) for i in nodes for j in nodes for k in vehicles if i!=j]


# model
model = gb.Model("VRPTW")

# decistion variables
x = model.addVars(arc_var, vtype=gb.GRB.INTEGER, name = 'x')
t = model.addVars(arc_times, vtype=gb.GRB.CONTINUOUS, name = 't')
D = model.addVars(arc_deliv, vtype = gb.GRB.INTEGER, name = 'D')   # Deliveries onboard arc i,j for vehicle k
P = model.addVars(arc_pickup, vtype = gb.GRB.INTEGER, name = 'P')   # Pickups onboard arc i,j for vehicle k

# objective 
model.setObjective(gb.quicksum(distances[i,j]*x[i,j,k] for i,j,k in arc_var),gb.GRB.MINIMIZE)

# constraints
# every island served exactly once
model.addConstrs(gb.quicksum(x[i,j,k] for i in nodes for k in vehicles if i!=j) == 1 for j in nodes)
model.addConstrs(gb.quicksum(x[i,j,k] for j in nodes for k in vehicles if i!=j) == 1 for i in nodes)

# === model.addConstrs(gb.quicksum(x[0,j,k] for j in clients) <= 1 for k in vehicles)
# === model.addConstrs(gb.quicksum(x[i,0,k] for i in clients) <= 1 for k in vehicles)

# more than one vehicle per node
# === model.addConstrs(gb.quicksum(x[i,j,k] for j in nodes for k in vehicles if i!=j) >=1 for i in clients)

# flow conservation

for j in nodes:
    counter1, counter2 = None, None
    for k in vehicles:    
        for i in nodes:
            if i!=j:
                if counter1 == None:
                    counter1 = D[i,j,k]    
                else:
                    counter1 += D[i,j,k]

                if counter2 == None:
                    counter2 = D[i,j,k]
                else:
                    counter2 += D[i,j,k]
                print(counter1,counter2)
    # if i!=j:
        # model.addConstr(counter1 - q[j] == counter2)
model.update()

# model.addConstrs(gb.quicksum(P[0,j,k] == 0) for j in nodes for k in vehicles if j!= 0)

# model.addConstrs(((gb.quicksum(D[i,j,k])-gb.quicksum(q[j]) for i in nodes for k in vehicles if i!=j) for j in nodes) == 
#                  gb.quicksum(D[j,i,k] for i in nodes for k in vehicles if i!=j) for j in nodes)
# model.addConstrs(gb.quicksum(gb.quicksum(P[i,j,k] for i in nodes for k in vehicles if i!=j) + b[j] for j in nodes) ==  
#                  gb.quicksum(P[j,i,k] for i in nodes for k in vehicles if i!=j) for j in nodes)



# === model.addConstrs(gb.quicksum(x[i,j,k] for j in nodes if i!=j)-gb.quicksum(x[j,i,k] for j in nodes if i!=j)==0 for i in nodes for k in vehicles)


# model.addConstrs(u[0,k] == Q[k] for k in vehicles)
# === model.addConstrs((x[i,j,k] == 1) >> (u[i,k] + dif[j] == u[j,k]) for i in clients for j in nodes for k in vehicles if i!=j)
# model.addConstrs((x[i,j,k] == 1 ) >>  (t[i,k]+s[i] + times[i,j] == t[j,k]) for i in clients for j in clients for k in vehicles if i!=j)



# vehicle capacity
# === model.addConstrs(gb.quicksum(q[i]*gb.quicksum(x[i,j,k] for j in nodes if i!= j) for i in clients) <= Q[k] for k in vehicles)
# === model.addConstrs(gb.quicksum(b[i]*gb.quicksum(x[i,j,k] for j in nodes if i!= j) for i in clients) <= Q[k] for k in vehicles)

# flow of time
# === model.addConstrs((x[i,j,k] == 1 ) >>  (t[i,k]+s[i] + times[i,j] == t[j,k]) for i in clients for j in clients for k in vehicles if i!=j)

# === model.addConstrs(t[i,k] >= e[i] for i,k in arc_times)
# === model.addConstrs(t[i,k] <= l[i] for i,k in arc_times)

# model.Params.timeLimit = 60
# model.Params.MIPGap = 0.1

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
    for i in nodes:
        if i!=0 and x[0,i,k].x > 0.99:
            aux=[0,i]
            while i!=0:
                j=i
                for h in nodes:
                    if j!=h and x[j,h,k].x>0.9:
                        aux.append(h)
                        i=h
            routes.append(aux)
            trucks.append(k)
# print(routes)
# print(trucks)

# calculate times
time_accum = list()
for n in range(len(routes)):
    for k in range(len(routes[n])-1):
        if k==0:
            aux=[0]
        else:
            i = routes[n][k]
            j = routes[n][k+1]
            t = times[i,j]+s[i]+aux[-1]
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
        plt.annotate('$q_{%d}=%d$ | $p_{%d}=%d$ | $t_{%d}=%d$'%(i,q[i],i,p[i],i,time_accum[r][n]),(X[i]+1,Y[i]))
    

patch = [mpatches.Patch(color=colorchoice[n],label="vehcile "+str(trucks[n])+"|cap="+str(Q[trucks[n]])) for n in range(len(trucks))]
plt.legend(handles=patch, loc='best')


plt.xlabel("Distance X")
plt.ylabel("Distance Y")
plt.title("VRP Solution")
plt.show()