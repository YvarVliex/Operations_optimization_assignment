# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:26:41 2021

@author: Nils de Krom
"""


import unittest
import numpy as np

from Azores_VR_program_v7 import Azores_VR


class MyTestCase(unittest.TestCase):
    
    def test_testcase_1(self):
           
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
        
        azor_case_1.plot_nodes_map()
        azor_case_1.plot_trajectories_map()
        
        
        ##Hand calc:
        HC_obj_AC = 2*(0.01591941*30.80584*80 + 2*92.8901 + 80*9.108) 
        HC_obj_AB = 2*(0.01591941*60.00833*80 + 2*92.8901 + 80*9.108) 
        
        HC_obj_tot = HC_obj_AB + HC_obj_AC
        
        self.assertAlmostEqual(azor_case_1.objective_value, HC_obj_tot, 3)
        
    def test_testcase_2(self):
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
        
        azor_case_2.plot_nodes_map()
        azor_case_2.plot_trajectories_map()
        
        
if __name__ == '__main__':
    unittest.main()


