import unittest
import numpy as np

from Azores_VR_program_v7 import Azores_VR

min_landingdist = 800
data_sheet = "Azores_Flight_Data_v4.xlsx"
txt_file = "coordinates_airports.txt"
data_distance_cols = "A:J"
data_deliv_cols = "B,D:L"
azor_test = Azores_VR(data_sheet, txt_file,min_landingdist, data_distance_cols, data_deliv_cols)


class MyTestCase(unittest.TestCase):
    
    # Test lengths of certain lists/dataframes/arrays
    def test_dataframe_lengths(self):
        self.assertAlmostEqual(len(azor_test.df_distance),9)
        self.assertAlmostEqual(len(azor_test.df_fleet),6)
        self.assertAlmostEqual(len(azor_test.df_cost),6)
        self.assertAlmostEqual(len(azor_test.df_deliv),1)
        self.assertAlmostEqual(len(azor_test.df_distance_2),9)
        self.assertAlmostEqual(len(azor_test.df_coordinates),9)
        self.assertAlmostEqual(azor_test.n_islands,9)
        self.assertAlmostEqual(len(azor_test.destinations),8)
        self.assertAlmostEqual(len(azor_test.nodes),9)
        self.assertAlmostEqual(len(azor_test.island_arcs),72)
        self.assertAlmostEqual(len(azor_test.q),9)
        self.assertAlmostEqual(len(azor_test.vehicles),6)
        self.assertAlmostEqual(len(azor_test.Q),6)
        self.assertAlmostEqual(azor_test.min_landingdist,800)
        self.assertAlmostEqual(len(azor_test.t_min),9)
        self.assertAlmostEqual(len(azor_test.t_max),9)

    # Test fleet values
    def test_fleet_values(self):
        self.assertTrue(azor_test.df_fleet.iloc[0,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_fleet.iloc[1,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_fleet.iloc[2,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_fleet.iloc[3,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_fleet.iloc[4,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_fleet.iloc[5,0], 'Bombardier Dash 8 Q400')
        
    # Test cost values
    def test_cost_values(self):    
        self.assertTrue(azor_test.df_cost.iloc[0,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_cost.iloc[1,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_cost.iloc[2,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_cost.iloc[3,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_cost.iloc[4,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_cost.iloc[5,0], 'Bombardier Dash 8 Q400')

    # Test demand values
    def test_demand_values(self):
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,0], 0)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,1], 32)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,2], 6)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,3], 22)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,4], 45)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,5], 0)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,6], 44)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,7], 51)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,8], 61)

    # Test distance matrix    
    def test_distance_2_matrix_values(self):
        self.distance_2_vals = np.matrix([[  0,    515,     508,    277,   256,        238,       252,       164,           98],
                                          [515,      0,      24,    243,   262,        277,       273,       362,          599],
                                          [508,     24,       0,    233,   253,        271,       270,       356,          589],
                                          [277,    243,     233,      0,    24,         50,        87,       144,          356],
                                          [256,    262,     253,     24,     0,         27,        70,       120,          337],
                                          [238,    277,     271,     50,    27,          0,        49,        95,          324],
                                          [252,    273,     270,     87,    70,         49,         0,        90,          344],
                                          [164,    362,     356,    144,   120,         95,        90,         0,          260],
                                          [ 98,    599,     589,    356,   337,        324,       344,       260,            0]])  
        for i in range(9):
            for j in range(9):
                self.assertAlmostEqual(azor_test.df_distance_2.iloc[i,j],self.distance_2_vals[i,j])

    # Test correct index per island and correct coordinates (X and Y)
    def test_coordinates(self):
        self.assertTrue(azor_test.df_coordinates.iloc[0,0], 'São Miguel')
        self.assertTrue(azor_test.df_coordinates.iloc[1,0], 'Corvo')
        self.assertTrue(azor_test.df_coordinates.iloc[2,0], 'Flores')
        self.assertTrue(azor_test.df_coordinates.iloc[3,0], 'Faial')
        self.assertTrue(azor_test.df_coordinates.iloc[4,0], 'Pico')
        self.assertTrue(azor_test.df_coordinates.iloc[5,0], 'São Jorge')
        self.assertTrue(azor_test.df_coordinates.iloc[6,0], 'Graciosa')
        self.assertTrue(azor_test.df_coordinates.iloc[7,0], 'Terceira')
        self.assertTrue(azor_test.df_coordinates.iloc[8,0], 'Santa Maria')

        self.assertAlmostEqual(azor_test.X[0], -25.710127, places=6)
        self.assertAlmostEqual(azor_test.X[1], -31.113625, places=6)
        self.assertAlmostEqual(azor_test.X[2], -31.132090, places=6)
        self.assertAlmostEqual(azor_test.X[3], -28.715015, places=6)
        self.assertAlmostEqual(azor_test.X[4], -28.441356, places=6)
        self.assertAlmostEqual(azor_test.X[5], -28.169865, places=6)
        self.assertAlmostEqual(azor_test.X[6], -28.028601, places=6)
        self.assertAlmostEqual(azor_test.X[7], -27.085298, places=6)
        self.assertAlmostEqual(azor_test.X[8], -25.170943, places=6)

        self.assertAlmostEqual(azor_test.Y[0],  37.745953, places=6)
        self.assertAlmostEqual(azor_test.Y[1],  39.671062, places=6)
        self.assertAlmostEqual(azor_test.Y[2],  39.458113, places=6)
        self.assertAlmostEqual(azor_test.Y[3],  38.519890, places=6)
        self.assertAlmostEqual(azor_test.Y[4],  38.554307, places=6)
        self.assertAlmostEqual(azor_test.Y[5],  38.663288, places=6)
        self.assertAlmostEqual(azor_test.Y[6],  39.092355, places=6)
        self.assertAlmostEqual(azor_test.Y[7],  38.757040, places=6)
        self.assertAlmostEqual(azor_test.Y[8],  36.973747, places=6)

    # Test velocity of fleet
    def test_fleet_velicty(self):
        azor_test.get_all_req_val()
        self.assertAlmostEqual(azor_test.k_Vcr_dct[0], 535)
        self.assertAlmostEqual(azor_test.k_Vcr_dct[1], 535)
        self.assertAlmostEqual(azor_test.k_Vcr_dct[2], 667)
        self.assertAlmostEqual(azor_test.k_Vcr_dct[3], 667)
        self.assertAlmostEqual(azor_test.k_Vcr_dct[4], 667)
        self.assertAlmostEqual(azor_test.k_Vcr_dct[5], 667)

    # Test time arrays for the different aircraft
    def test_time_arrays(self):
        azor_test.get_all_req_val()
        self.assertAlmostEqual(len(azor_test.time_df_dct),6)
        # Test lengths of each of the aircraft time arrays
        for i in range(6):
            self.assertAlmostEqual(len(azor_test.time_df_dct[i]),9)
        # Test if the two Q200 aircraft have same values
        for j in range(9):
            for k in range(9):
                self.assertAlmostEqual(azor_test.time_df_dct[0].iloc[j,k],azor_test.time_df_dct[1].iloc[j,k])
        # Test if the four Q400 aircraft have same values
        for j in range(9):
            for k in range(9):
                self.assertAlmostEqual(azor_test.time_df_dct[2].iloc[j,k],azor_test.time_df_dct[3].iloc[j,k])
        for j in range(9):
            for k in range(9):
                self.assertAlmostEqual(azor_test.time_df_dct[2].iloc[j,k],azor_test.time_df_dct[4].iloc[j,k])
        for j in range(9):
            for k in range(9):
                self.assertAlmostEqual(azor_test.time_df_dct[2].iloc[j,k],azor_test.time_df_dct[5].iloc[j,k])

    # Test if distances are correctly parsed (for usage in GUROBI Model) for all i,j combinations
    def test_distance_arc_vars(self):
        azor_test.get_all_req_val()
        self.assertAlmostEqual(len(azor_test.distances),72)
        for i in range(9):
            for j in range(9):
                if i!=j:
                    self.assertAlmostEqual(azor_test.distances[(i,j)],azor_test.df_distance_2.iloc[i,j])

    # Test if times are correctly parsed (for usage in GUROBI Model) for all i,j,k combinations
    def test_times_arc_vars(self):
        azor_test.get_all_req_val()
        self.assertAlmostEqual(len(azor_test.times),432)
        for i in range(9):
            for j in range(9):
                if i!=j:
                    for k in range(6):
                        self.assertAlmostEqual(azor_test.times[(i,j,k)],azor_test.time_df_dct[k].iloc[i,j])

    # Tests if the cost value (Obj. Value) gives the correct value, compared with hand calculation of all cost components,
    # taking into account the final solution the GUROBI solver gives for the trips
    def test_objective_function_outcome(self):

        # Values from GUROBI solver
        self.trajectories_Q200 = [[0, 2, 3, 0], [0, 1, 0]]
        self.trajectories_Q400 = [[0, 8, 0], [0, 7, 0], [0, 6, 0], [0, 4, 0]]
        self.dist_Q200 = 0
        self.dist_Q400 = 0
        # Get distances flown for each aircraft type
        for i in self.trajectories_Q200:
            for j in range(len(i)-1):
                self.dist_Q200 += azor_test.df_distance_2.iloc[i[j],i[j+1]]
        for i in self.trajectories_Q400:
            for j in range(len(i)-1):
                self.dist_Q400 += azor_test.df_distance_2.iloc[i[j],i[j+1]]

        # Values from excel database file
        self.fuelcost_Q200_perkm_perPAX = 0.024071414
        self.fuelcost_Q400_perkm_perPAX = 0.01591941
        self.Q_Q200 = 37
        self.Q_Q400 = 80

        # Get total fuel cost for each aircraft type
        self.fuelcost_Q200_total = self.fuelcost_Q200_perkm_perPAX * self.Q_Q200 * self.dist_Q200
        self.fuelcost_Q400_total = self.fuelcost_Q400_perkm_perPAX * self.Q_Q400 * self.dist_Q400

        # Get number of flights per aircraft type
        self.nr_flights_total_Q200 = 0
        self.nr_flights_total_Q400 = 0
        for i in self.trajectories_Q200:
            self.nr_flights_total_Q200 += (len(i)-1)
        for i in self.trajectories_Q400:
            self.nr_flights_total_Q400 += (len(i)-1)
        
        # Values from excel database file
        self.cost_TO_LA_Q200 = 51.95468
        self.cost_TO_LA_Q400 = 92.8901
        self.cost_perPAX = 9.108

        # Get total take-off and landing costs for each aircraft type
        self.TO_LA_cost_Q200_total = self.cost_TO_LA_Q200 * 2 * self.nr_flights_total_Q200
        self.TO_LA_cost_Q400_total = self.cost_TO_LA_Q400 * 2 * self.nr_flights_total_Q400

        # Get total passenger costs for each aircraft type
        self.PAX_cost_Q200_total = self.nr_flights_total_Q200 * self.Q_Q200 * self.cost_perPAX
        self.PAX_cost_Q400_total = self.nr_flights_total_Q400 * self.Q_Q400 * self.cost_perPAX

        # Add all costs together, run the AZmodel file and compare the values
        self.total_costs = self.fuelcost_Q200_total + self.fuelcost_Q400_total + self.TO_LA_cost_Q200_total + self.TO_LA_cost_Q400_total + self.PAX_cost_Q200_total + self.PAX_cost_Q400_total
        azor_test.get_all_req_val()
        azor_test.initialise_model()
        azor_test.adding_constraints()
        azor_test.get_solution()
        self.assertAlmostEqual(azor_test.AZmodel.ObjVal, self.total_costs, places=4)


if __name__ == '__main__':
    unittest.main()
