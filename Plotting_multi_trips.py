# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 11:43:09 2021

@author: Nils de Krom
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

def txt_file_reader(txt_file, col_indx):
    return pd.read_csv(txt_file, index_col = col_indx)

def excel_data_obtainer(filename, sheetname, start_row, end_row, cols):
    df_temp = pd.read_excel(filename, sheetname, usecols = cols, skiprows = start_row, nrows= (end_row-start_row) )
    return df_temp

txt_file = "coordinates_airports.txt"
data_sheet = "Azores_Flight_Data_v4.xlsx"
data_deliv_cols = "B,D:L"

df_deliv = excel_data_obtainer(data_sheet, "Demand Table", 0, 2, data_deliv_cols).drop(0).set_index("Start").astype('float64').round(0)

df_coordinates = txt_file_reader(txt_file, 0).reindex(df_deliv.columns)

X = df_coordinates["x"]
Y = df_coordinates["y"]


# routes = [[[0,2,0],3],
#           [[0,3,0],12],
#           [[0,4,0],11],
#           [[0,5,0],7],
#           [[0,6,0],3],
#           [[0,7,0],46],
#           [[0,8,0],4]]

# routes_ar = np.array([[0,2,0],
#                       [0,3,0],
#                       [0,4,0],
#                       [0,5,0],
#                       [0,6,0],
#                       [0,7,0],
#                       [0,8,0]])

routes_ar = [[0,2,0],
            [0,3,0],
            [0,4,0],
            [0,5,0],
            [0,6,0],
            [0,7,0],
            [0,8,0]]

routes_nf = [3,12,11,7,3,46,4]

# colorchoice = ['red', 'green', 'black', 'grey', 'skyblue', 'orange','gold']  


mymap = matplotlib.colors.LinearSegmentedColormap.from_list('mycolors',['yellow','red'])
Z = [[0,0],[0,0]]
levels = range(0,max(routes_nf)+1,1)
CS3 = plt.contourf(Z, levels, cmap=mymap)
plt.clf()


for r in range(len(routes_ar)):
    for n in range(len(routes_ar[r])-1):
        i = routes_ar[r][n]
        j = routes_ar[r][n+1]
               
        p = (routes_nf[r]-min(routes_nf))/(max(routes_nf)-min(routes_nf))
        r = 1
        g = 1-p
        b = 0
        plt.plot([X[i],X[j]],[Y[i],Y[j]], color=(r,g,b), zorder=0)
   

plt.scatter(X, Y)#, label=airports)
plt.gca().set_aspect('equal', adjustable='box')
for i in range(len(df_coordinates)):
    plt.annotate(f"$n_{i}$", (X[i], Y[i]), xytext = (X[i]-0.01, Y[i]-0.1))
    
for r in range(len(routes_ar)):
    for n in range(len(routes_ar[r])-1):
        i = routes_ar[r][n]
        j = routes_ar[r][n+1]
        
        middle_x = (X[j]-X[i])*0.5+X[i]
        middle_y = (Y[j]-Y[i])*0.5+Y[i]
        
        plt.annotate(f"{routes_nf[r]}", (middle_x, middle_y), xytext = (middle_x+0.01, middle_y-0.04))
        
        

plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.title('Flown Direct Flights')
plt.colorbar(CS3)
plt.grid()
plt.show()
        


