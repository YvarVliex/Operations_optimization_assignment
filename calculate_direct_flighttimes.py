from datetime import time
import numpy as np

from Azores_VR_program_v7 import Azores_VR

min_landingdist = 800
data_sheet = "Azores_Flight_Data_v4.xlsx"
txt_file = "coordinates_airports.txt"
data_distance_cols = "A:J"
data_deliv_cols = "B,D:L"
azor_test = Azores_VR(data_sheet, txt_file,min_landingdist, data_distance_cols, data_deliv_cols)

azor_test.get_all_req_val()

main_routes = []

# Sao Miguel is hub
#nr_direct = [0,0,3,12,11,7,3,46,4]

# Terceira is hub
nr_direct = [116, 0, 3, 12, 11, 7, 3, 0, 4]

print(len(nr_direct))

for l in range(len(nr_direct)):
    for i in range(nr_direct[l]):
        # Sao Miguel is hub
        #main_routes.append(0)
        #main_routes.append(l)

        # Terceira is hub
        main_routes.append(7)
        main_routes.append(l)

#main_routes.append(0)    
main_routes.append(7)

print(main_routes)
time_counter = 0
for i in range(len(main_routes)-1):
    print(azor_test.times[main_routes[i],main_routes[i+1],2])
    time_counter += (azor_test.times[main_routes[i],main_routes[i+1],2])

# time in hourse
time_counter_hours = time_counter / 60
time_counter_hours_per_day_per_ac = time_counter / 60 / 6.5 / 4

print("Time required total hours")
print(time_counter_hours)

print("Time required hours per day per ac")
print(time_counter_hours_per_day_per_ac)
