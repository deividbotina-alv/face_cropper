'''
1)This script thakes one folder with faces of one dataset, and it synchronize the faces with
their respective ground truth, by means of the alignment between rPPG signals and GT files.
2)Ground truth files are saved again but now with all postprocessing done, i.e. resampled to
25 Hz, detrended, smoothed and normalized between [-1,1]
3)Additionally, it removes the problematic parts of every subject based in a given excel file
with the reliables part of the signal e.g. In subject FT001 of 50 seconds long, just take the
rank between 20-50 because the ground truth signal acquisition failed in first 20 seconds
'''

#%% GLOBAL VARIABLES
PathL_Face = r'J:\faces\original\MMSE'# Path to load dataset with faces
PathL_GT = r'J:\faces\original\MMSE'# Path to load dataset with ground truth files
PathL_rPPG = r'J:\POS_traces\MMSE-HR\filter'# Path to load dataset with rPPG files
PathS_Faces = r'J:\faces\synchronized\MMSE'# Path to save faces aligned
MinimumSizeVideoInSeconds = 15 # Ouputs with duration less than this value will be ignored
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
import pandas as pd
import scipy.io as sio

from ubfc_functions import resample_by_interpolation, smooth, detrendsignal, pltnow, normalize, butter_bandpass_filter, phase_align

#%% CLASSES AND FUNCTIONS

class synchronize_faces():
    # CONSTRUCTOR
    def __init__(self,PathL_Face,PathL_GT,PathL_rPPG,PathS_Faces,MinLengthInSec,dataset):
        self.PathL_Face = PathL_Face # Path to load dataset with faces (1 folder per subject)
        self.PathL_GT = PathL_GT # Path to load dataset with ground truth files (1 file per subject)
        self.PathL_rPPG = PathL_rPPG # Path to load dataset with rPPG files (1 file per subject)
        self.PathS_Faces = PathS_Faces # Path to save faces aligned with GT file
        self.MinLengthInSec = MinLengthInSec # Minimum length in seconds for a subject, if it is less than this value, ignore the subject
        self.dataset = dataset
        #Initialize parameters for each dataset
        if dataset.upper()=='MMSE':
            self.fr_gt = 1000 # frequency ground truth in Hz
            self.fr_rppg = 25 # frequency/fps rPPG/frames
        elif dataset.upper()=='VIPL':
            pass
        elif dataset.upper()=='COHFACE':
            pass
        elif dataset.upper()=='PURE':
            pass

    # FUNCTION TO REPORT SOMETHING IN A .TXT FILE
    def report_TXT(self,nameFile,message):
        fileTXT = open(nameFile,"a")
        fileTXT.write(message)
        fileTXT.close()
        
    # FUCNTION TO SAVE ARRAY IN TXT FILE
    def saveArray_TXT(self,nameFile,array):
        a_file = open(nameFile, "w")
        for row in array:
            np.savetxt(a_file, row)
        a_file.close()

    # FUNCTION TO FIND LIST OF SUBJECTS IN self.PathL_Face 
    def find_faces_folders(self):
        self.faces_list = []
        directory_contents = os.listdir(self.PathL_Face)
        for item in directory_contents:
            if os.path.isdir(join(self.PathL_Face,item)):#Take only the folder names
                self.faces_list.append(join(self.PathL_Face,item))
        self.faces_list = np.array(natsorted(self.faces_list))
        
    # FUNCTION TO FIND GROUND TRUTH FILES in self.PathL_GT
    def find_GT_files(self):
        self.GT_list = []
        for root, dirs, files in os.walk(self.PathL_GT):
            for name in files:
                if name.split('.')[0].endswith(('_gt','_GT')):
                    self.GT_list.append(join(root,name))
        self.GT_list = np.array(natsorted(self.GT_list))
        
    # FUNCTION TO FIND RPPG FILES in self.PathL_rPPG
    def find_rPPG_files(self):
        self.rPPG_list = []
        for root, dirs, files in os.walk(self.PathL_rPPG):
            for name in files:
                if name.split('.')[0].endswith('_rppg_POS'):
                    self.rPPG_list.append(join(root,name))
        self.rPPG_list = np.array(natsorted(self.rPPG_list))
        
    # FLAG REFERING TO WETHER THE NUMBER OF GROUND TRUTH FILES, FACES FOLDERS, AND RPPG FILES ARE THE SAME
    def same_number_of_faces_GT_rPPG(self):
        return np.size(self.faces_list) == np.size(self.GT_list) == np.size(self.rPPG_list)
    
    # FUNCTION TO LOAD GT FILE + POSPROCESSING
    def load_GT(self,path):
        if self.dataset.upper()=='MMSE':
            #Load file
            gt = np.loadtxt(path); #pltnow(gt,val=1,fr=1000)
            # Resample GT from 1000 Hz to 25 Hz (rPPG sample frequency)
            gt = resample_by_interpolation(gt, input_fs=self.fr_gt, output_fs=25)# pltnow(gt,val=1,fr=25)
            # Center ground truth in 0
            gt = gt-np.mean(gt)# pltnow(gt,val=1,fr=25)
            # Detrend signal
            gt = detrendsignal(gt) # pltnow(gt,val=1,fr=25)
            # smooth the signal
            gt = smooth(gt, window_len=5, window='flat') # pltnow(gt,val=1,fr=25)
            # Normalize between [-1,1]
            gt = normalize(gt);gt = np.resize(gt,gt.size) # pltnow(gt,val=1,fr=25)
            
        elif self.dataset.upper()=='VIPL':
            pass
        elif self.dataset.upper()=='COHFACE':
            pass
        elif self.dataset.upper()=='PURE':
            pass
        
        return gt
    
    # FUNCTION TO LOAD RPPG FILE + POSPROCESSING
    def load_rPPG(self,path):
        m = sio.loadmat(path)
        t = m['timeTrace'];t=np.resize(t,(t.shape[1],))
        pulseTrace = m['pulseTrace']; pulseTrace=np.resize(pulseTrace,(pulseTrace.shape[1],))#pltnow(pulseTrace,val=1,fr=25)
        # Center ground truth in 0
        pulseTrace = pulseTrace-np.mean(pulseTrace)# pltnow(pulseTrace,val=1,fr=25)
        # Detrend signal
        pulseTrace = detrendsignal(pulseTrace) # pltnow(pulseTrace,val=1,fr=25)
        # smooth the signal
        pulseTrace = smooth(pulseTrace, window_len=5, window='flat') # pltnow(pulseTrace,val=1,fr=25)
        # Band Pass filter and re-aligment because after bandpass the signal moves a little bit
        # BP = butter_bandpass_filter(pulseTrace, lowcut=0.7, highcut=3.5, fs=25, order=8)
        # ROI_ini = 0 # begin in 0 seconds
        # ROI_end = ROI_ini+(5/0.04) # finish 5 seconds latter        
        # s2 = phase_align(pulseTrace, BP, [ROI_ini,ROI_end])# align by phase
        # pulseTrace = np.roll(BP, int(s2))#
        
        # Normalize between [-1,1]
        pulseTrace = normalize(pulseTrace);pulseTrace = np.resize(pulseTrace,pulseTrace.size) # pltnow(pulseTrace,val=1,fr=25)
        return pulseTrace
    
    # FUNCTION TO LOAD THE INDEX OF THE FRAMES OF ONE FACES FOLDER
    def load_faces_index(self,path):
        faces_idx = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith(('.png')):
                    faces_idx.append(name)
        return np.array(natsorted(faces_idx))

    # FUNCTION TO GET THE NEW NAMES OF FRAMES SYNCHRONIZED
    def SincronizameEsta(self,name,rPPG,GT,faces_idx,cuting):
        flag_Valid = True # Valid subject flag
        # name = name of current subject
        # cuting = dataframe with the names of the subjects with the position of the reliable segments
        Begin_in_sec = 0 # Begin of reliable segment in seconds
        End_in_sec = -1 # End of reliable segment in seconds
        # If is the case, find the begin and the end of the reliable segment for this subject
        for i in range(0,len(cuting)):
            if cuting['Subject'][i]==name:
                Begin_in_sec = cuting['Begin[s]'][i]
                End_in_sec = cuting['End[s]'][i]
                break
            
        # rPPG and GT should have same length. Cut last part of the longest one
        if rPPG.size < GT.size:# if rppg is shorter than gt
            GT = GT[0:rPPG.size]
        else:# if gt is shorter than rppg
            rPPG = rPPG[0:GT.size]
            faces_idx = faces_idx[0:GT.size]
        #pltnow(rPPG,GT,val=2,fr=25)
        
        # Take only the reliable segment
        self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),'Reliable segment in [{},{}] s.\n'.format(Begin_in_sec,End_in_sec))
        print('Reliable segment in [{},{}] s.'.format(Begin_in_sec,End_in_sec))
        cut_ini = int(Begin_in_sec*self.fr_rppg)
        cut_end = int(End_in_sec*self.fr_rppg)
        if cut_end < 0: # If is negative value it means in Excel file End_in_sec was -1 (last position)
            GT = GT[cut_ini:]
            rPPG = rPPG[cut_ini:]
            faces_idx = faces_idx[cut_ini:]
        else: # If End_in_sec was different than -1
            GT = GT[cut_ini:cut_end]
            rPPG = rPPG[cut_ini:cut_end]
            faces_idx = faces_idx[cut_ini:cut_end]
        #pltnow(rPPG,GT,val=2,fr=25)
        
        if rPPG.size*1/self.fr_rppg<self.MinLengthInSec:#If signals is less than 5 seconds:
            self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),'[ERROR]: size is {}, (less than {} seconds)\n'.format(rPPG.size*1/self.fr_rppg,self.MinLengthInSec))
            print('[ERROR]: size is {}, (less than {} seconds)'.format(rPPG.size*1/self.fr_rppg,self.MinLengthInSec))
            faces_idx = []
            new_faces_idx = []
            flag_Valid = False
        else:
            # Synchronize rPPG in GT, the same shift must be aplied in faces_idx
            ROI_ini = 0*self.fr_rppg # begin in first 0 seconds
            ROI_end = ROI_ini+(5*self.fr_rppg)#finish 5 seconds latter
            #pltnow(GT[int(ROI_ini):int(ROI_end)],rPPG[int(ROI_ini):int(ROI_end)],2)
            s1 = phase_align(GT, rPPG, [ROI_ini,ROI_end])# align by phase
            rPPG = np.roll(rPPG, int(s1)) # pltnow(rPPG,GT,val=2,fr=25)
            new_faces_idx = np.roll(faces_idx, int(s1)) # we apply the same shift in frames
            self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),'Shifted {:0.2f} frames, ({:0.2f} seconds)\n'.format(s1,s1*1/self.fr_rppg))
            print('Shifted {:0.2f} frames, ({:0.2f} seconds)'.format(s1,s1*1/self.fr_rppg))

        return faces_idx,new_faces_idx,GT,flag_Valid #GT may changed

    # FUNCTION TO SYNCHRONIZE FACES WITH GROUND TRUTH, USING ALINMENT BETWEEN RPPG AND GT FILES
    # ALSO A SELECTION OF ONLY RELIABLE PARTS OF EACH SUBJECT BASED IN THE GROUND TRUTH IS DONE.
    def TakeOnlyReliableSegmentAndSynchronize(self,PathL_ExcelFile):
        # PathL_ExcelFile: Path with the excel file with the subject names and the reliable segments of them
        # Load Excel file with reliable segments of each subject
        cuting = pd.read_excel(PathL_ExcelFile)
        # Start report of current dataset
        self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),
                        "FACE ALIGNMENT WITH GROUND TRUTH BY MEANS OF THE RPPG-GT ALIGNMENT: INPUT DATA IN {}\n".format(self.PathL_Face))
        for i in range(0,len(self.faces_list)): # For all subjects
            name_subject_face = self.faces_list[i].split(os.path.sep)[-1]
            name_subject_GT = self.GT_list[i].split(os.path.sep)[-1].split('_gt')[0]
            name_subject_rppg = self.rPPG_list[i].split(os.path.sep)[-1].split('_rppg_POS')[0]

            # face frames, GT file, and rPPG file must correspond to the same subject
            if name_subject_face == name_subject_GT == name_subject_rppg:
                name = name_subject_face # general subject name
                # Does the current subject output folder exist? if so, skip to the next subject
                if not os.path.exists(join(self.PathS_Faces,name)):
                    os.makedirs(join(self.PathS_Faces,name))
                else:
                    continue
                print(name)
                self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),'\n'+name+'\n')
                faces_idx = self.load_faces_index(self.faces_list[i]) # Get index of frames in current subject
                rPPG = self.load_rPPG(self.rPPG_list[i]) # Load rPPG ready to be compared
                # For some reason the POS-rPPG signals has one less value so we duplicate the last one in order to have same length.
                rPPG = np.concatenate((rPPG,(rPPG[-1],)),axis=0)
                # number of frames in "faces_idx" must be the same found in "rPPG" since rPPG was taken from the same video
                if len(rPPG)==len(faces_idx):
                    GT = self.load_GT(self.GT_list[i]) # Load GT signal ready to be compared. pltnow(GT,rPPG,val=3,fr=25)
                    old_idx,new_idx,newGT,valid = self.SincronizameEsta(name,rPPG,GT,faces_idx,cuting)
                    if valid: # If subject had a valid length, save it.
                        for j in range(0,len(old_idx)):
                            old_frame = cv2.imread(join(self.PathL_Face,name,new_idx[j]))
                            cv2.imwrite(join(self.PathS_Faces,name,old_idx[j]), old_frame)
                        # Finally save new GT file because it may changed
                        self.saveArray_TXT(join(self.PathS_Faces,name,name+'_gt.txt'),[newGT])
                else:
                    self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),
                        "[ERROR]: rPPG length = {} and number of frames = {}. They should be the same\n".format(len(rPPG),len(faces_idx)))
                    print('{} has different length in rPPG and frames, skipping.'.format(name))
                    continue

            else:
                print('Files for Subject {} do not correspond. Face:{},GT:{},rppg:{}'.format(i,name_subject_face,name_subject_GT,name_subject_rppg))
                continue
            

        


#%% MAIN

MMSE = synchronize_faces(PathL_Face,PathL_GT,PathL_rPPG,PathS_Faces,MinimumSizeVideoInSeconds,'MMSE')
MMSE.find_faces_folders()
MMSE.find_GT_files()
MMSE.find_rPPG_files()
if MMSE.same_number_of_faces_GT_rPPG():
    MMSE.TakeOnlyReliableSegmentAndSynchronize(r'E:\repos\face_cropper\source\removeGT\GT_SignalsToCut_MMSE.xlsx')
else:
    print('Error, different number of files in faces folders, GT files and/or rPPG files')