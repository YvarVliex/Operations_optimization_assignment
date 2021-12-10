# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:26:41 2021

@author: Nils de Krom
"""


import unittest
from Azores_VR_program_v7 import Azores_VR


class MyTestCase(unittest.TestCase):
    
    def test_testcase_1(self):
        "Test that checks if the program will provide same output as hand determined solution"
        data_sheet = "Test_cases/Test_case_1.xlsx"
        txt_file = "Test_cases/Test_case_1_coords.txt"
        min_landingdist = 800
        
        data_distance_cols_case_1 = "A:D"
        data_deliv_cols_case_1 = "B,D:F"
        azor_case_1 = Azores_VR(data_sheet, txt_file,min_landingdist, data_distance_cols_case_1, data_deliv_cols_case_1, "no")
        
        azor_case_1.get_all_req_val()
        azor_case_1.initialise_model()
        azor_case_1.adding_constraints()
        
        azor_case_1.get_solution()  
        
        # azor_case_1.plot_nodes_map()
        azor_case_1.plot_trajectories_map()
        
        
        ##Hand calc:
        HC_obj_AC = 2*(0.01591941*30.80584*80 + 2*92.8901 + 80*9.108) 
        HC_obj_AB = 2*(0.01591941*60.00833*80 + 2*92.8901 + 80*9.108) 
        
        HC_obj_tot = HC_obj_AB + HC_obj_AC
        
        self.assertAlmostEqual(azor_case_1.objective_value, HC_obj_tot, 3)
        
    def test_testcase_2(self):
        "Test that checks if the program will provide same output as hand determined solution"
        
        data_sheet = "Test_cases/Test_case_2.xlsx"
        txt_file = "Test_cases/Test_case_1_coords.txt"
        min_landingdist = 800
        
        data_distance_cols_case_2 = "A:D"
        data_deliv_cols_case_2 = "B,D:F"
        azor_case_2 = Azores_VR(data_sheet, txt_file,min_landingdist, data_distance_cols_case_2, data_deliv_cols_case_2, "no")
        
        azor_case_2.get_all_req_val()
        azor_case_2.initialise_model()
        azor_case_2.adding_constraints()
        
        azor_case_2.get_solution()  
        
        # azor_case_2.plot_nodes_map()
        azor_case_2.plot_trajectories_map()
        
        threshold = 0.01 #%
        
        #hand calc
        
        #Correct option
        HC_obj_BC = 2*(0.01591941*30.59411708*80 + 2*92.8901 + 80*9.108)
        HC_obj_BA = 2*(0.0240714136648884*60.00833*37 + 2*51.95468 + 37*9.108)
        
        perc_diff_1 = (azor_case_2.objective_value - (HC_obj_BC+HC_obj_BA))/(HC_obj_BC+HC_obj_BA) * 100
        perc_diff_2 = (azor_case_2.objective_value - (HC_obj_BC+HC_obj_BA))/(azor_case_2.objective_value) * 100
        
        #Incorrect option
        HC_obj_BC_inc = (0.01591941*30.59411708*80 + 2*92.8901 + 80*9.108)
        HC_obj_CA_inc = (0.01591941*30.80584*80 + 2*92.8901 + 80*9.108)
        HC_obj_AB_inc = (0.01591941*60.00833*80 + 2*92.8901 + 80*9.108)
        
        HC_obj_more_exp = HC_obj_BC_inc+HC_obj_CA_inc+HC_obj_AB_inc
        
        self.assertLess(perc_diff_1, threshold)
        self.assertLess(perc_diff_2, threshold)
        
        self.assertLess(azor_case_2.objective_value, HC_obj_more_exp)
        
        # print('############################')
        # print(azor_case_2.aircrafts)
        
    def test_testcase_3(self):
        "Test that checks if the program will take the slightly cheaper versions of the vehicles"
        data_sheet = "Test_cases/Test_case_3.xlsx"
        txt_file = "Test_cases/Test_case_1_coords.txt"
        min_landingdist = 800
        
        data_distance_cols_case_1 = "A:D"
        data_deliv_cols_case_1 = "B,D:F"
        azor_case_3 = Azores_VR(data_sheet, txt_file,min_landingdist, data_distance_cols_case_1, data_deliv_cols_case_1, "no")
        
        azor_case_3.get_all_req_val()
        azor_case_3.initialise_model()
        azor_case_3.adding_constraints()
        
        azor_case_3.get_solution()  
        
        # azor_case_1.plot_nodes_map()
        azor_case_3.plot_trajectories_map()
        
        
        ##Hand check
        cheaper_vehicles = [2,3]
        
        self.assertEqual(azor_case_3.aircrafts, cheaper_vehicles)
        
if __name__ == '__main__':
    unittest.main()


