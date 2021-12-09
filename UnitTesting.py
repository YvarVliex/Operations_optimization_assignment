import unittest
import numpy as np

from Azores_VR_program_v7 import Azores_VR

min_landingdist = 800
data_sheet = "Azores_Flight_Data_v4.xlsx"
txt_file = "coordinates_airports.txt"
azor_test = Azores_VR(data_sheet, txt_file,min_landingdist)


class MyTestCase(unittest.TestCase):
    
    def test_dataframe_lengths(self):
        self.assertAlmostEqual(len(azor_test.df_distance),9)
        self.assertAlmostEqual(len(azor_test.df_fleet),6)
        self.assertAlmostEqual(len(azor_test.df_cost),6)
        self.assertAlmostEqual(len(azor_test.df_deliv),1)
        self.assertAlmostEqual(len(azor_test.df_distance_2),9)
        self.assertAlmostEqual(len(azor_test.df_coordinates),9)

    def test_dataframe_values(self):
        self.assertTrue(azor_test.df_fleet.iloc[0,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_fleet.iloc[1,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_fleet.iloc[2,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_fleet.iloc[3,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_fleet.iloc[4,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_fleet.iloc[5,0], 'Bombardier Dash 8 Q400')
        
        self.assertTrue(azor_test.df_cost.iloc[0,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_cost.iloc[1,0], 'Bombardier Dash 8 Q200')
        self.assertTrue(azor_test.df_cost.iloc[2,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_cost.iloc[3,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_cost.iloc[4,0], 'Bombardier Dash 8 Q400')
        self.assertTrue(azor_test.df_cost.iloc[5,0], 'Bombardier Dash 8 Q400')

        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,0], 0)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,1], 32)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,2], 6)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,3], 22)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,4], 45)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,5], 0)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,6], 44)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,7], 51)
        self.assertAlmostEqual(azor_test.df_deliv.iloc[0,8], 61)
        
    def test_distance_2_matrix_values(self):
        distance_2_vals = np.matrix([[  0,    515,     508,    277,   256,        238,       252,       164,           98],
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
                self.assertAlmostEqual(azor_test.df_distance_2.iloc[i,j],distance_2_vals[i,j])

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


if __name__ == '__main__':
    unittest.main()
