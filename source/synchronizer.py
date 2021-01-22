'''
This script thakes one folder with faces of one dataset, and it synchronize the faces with
their respective ground truth, by means of the alignment between rPPG signals and GT files.
Additionally, it removes the problematic parts of every subject based in a given excel file
with the reliables part of the signal e.g. In subject FT001 of 50 seconds long, just take the
rank between 20-50 because the ground truth signal acquisition failed in first 20 seconds
'''

#%% GLOBAL VARIABLES
PathL_Face = r'J:\faces\original\MMSE'# Path to load dataset with faces
PathL_GT = r'J:\faces\original\MMSE'# Path to load dataset with ground truth files
PathL_rPPG = r'J:\POS_traces\MMSE-HR\filter'# Path to load dataset with rPPG files

PathS_Faces = r'J:\faces\synchronized\MMSE'# Path to save faces aligned

#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from os import listdir
from os.path import join
import shutil as sh
from natsort import natsorted
import glob
import sys
import cv2
import copy

#%% CLASSES AND FUNCTIONS

class synchronize_faces():
    # CONSTRUCTOR
    def __inint__(self,PathL_Face,PathL_GT,PathL_rPPG,PathS_Faces):
        self.PathL_Face = PathL_Face # Path to load dataset with faces (1 folder per subject)
        self.PathL_GT = PathL_GT # Path to load dataset with ground truth files (1 file per subject)
        self.PathL_rPPG = PathL_rPPG # Path to load dataset with rPPG files (1 file per subject)
        self.PathS_Faces = PathS_Faces # Path to save faces aligned with GT file
        
    # FUNCTION TO FIND LIST OF SUBJECTS IN PathL_Face 
    def find_faces(self):
        faces_list = []
        for root, dirs, files in os.walk(self.PathL_Face):
            faces_list.append(dirs)
        return faces_list
        

#%% MAIN

syncrohize_MMSE = synchronize_faces(PathL_Face,PathL_GT,PathL_rPPG,PathS_Faces)
#