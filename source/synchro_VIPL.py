'''
1)This script thakes one folder with faces of one dataset, and it synchronize the faces with
their respective ground truth, by means of the alignment between rPPG signals and GT files.
2)Ground truth files are saved again but now with all postprocessing done, i.e. resampled to
25 Hz, detrended, smoothed and normalized between [-1,1]
3)Additionally, it removes the problematic parts of every subject based in a given excel file
with the reliables part of the signal e.g. In subject FT001 of 50 seconds long, just take the
rank between 20-50 because the ground truth signal acquisition failed in first 20 seconds
4) VIPL source 1 and 2 have a timestamp.txt file (in ms) with the frame rate of the videos,
source did not have the timestamp file but the videos in this source had 30fps. Thus, when I ran the
script "VIPLHR_run_pvm.py" i also create a fake "timestam.txt" for source 3 with values as follows:
[0,1/30,2(1/30),3(1/30)....len(rPPG)*(1/30)] and I saved the rPPG with the GT and timestam in PathL_rPPG.
This way in this current scriptm I can load a "timesamp.txt" file for all subjects: source 1, source 2, source 3.
But originally only source 1 and 3 had these files. At the output of this script I also saved timestamp for all sources
but now in seconds instead of miliseconds.
'''

#%% GLOBAL VARIABLES
PathL_Face = r'J:\faces\128_128\original\VIPL'# Path to load dataset with faces
PathL_GT = r'J:\faces\128_128\original\VIPL'# Path to load dataset with ground truth files
PathL_rPPG = r'J:\PVM_traces\nofilter\VIPL-HR'# Path to load dataset with rPPG files
PathS_Faces = r'J:\faces\128_128\synchronized\VIPL'# Path to save faces aligned
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
from scipy.interpolate import interp1d

from ubfc_functions import resample_by_interpolation, smooth, detrendsignal, pltnow, normalize, butter_bandpass_filter, phase_align
#%% CLASSES AND FUNCTIONS

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

class synchronize_faces():
    # CONSTRUCTOR
    def __init__(self,PathL_Face,PathL_GT,PathL_rPPG,PathS_Faces,MinLengthInSec,dataset):
        self.PathL_Face = PathL_Face # Path to load dataset with faces (1 folder per subject)
        self.PathL_GT = PathL_GT # Path to load dataset with ground truth files (1 file per subject)
        self.PathL_rPPG = PathL_rPPG # Path to load dataset with rPPG files (1 file per subject)
        self.PathS_Faces = PathS_Faces # Path to save faces aligned with GT file
        self.MinLengthInSec = MinLengthInSec # Minimum length in seconds for a subject, if it is less than this value, ignore the subject
        self.dataset = dataset.upper()
        #Initialize parameters for each dataset
        if self.dataset=='MMSE':
            self.fr_gt = 1000 # frequency ground truth in Hz
            self.fr_rppg = 25 # frequency/fps rPPG/frames
        elif self.dataset=='VIPL':
            self.fr_gt = 60 # frequency ground truth in Hz
            #For VIPL source 1 and 2 have a timestamp.txt file with the frame rate
            #source 2 does not have but it works at 30fps so, in PathL_rPPG I create a fake "timesamp.txt"
            #file wiht values [0,1/30,2(1/30),3(1/30)....len(rPPG)*(1/30)]. This way in this script I can
            #Load a "timesamp.txt" file for all subjects: source 1, source 2, source 3.
        elif self.dataset=='COHFACE':
            pass
        elif self.dataset=='PURE':
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
                if name.split('.')[0].endswith('_rppg'):
                    self.rPPG_list.append(join(root,name))
        self.rPPG_list = np.array(natsorted(self.rPPG_list))

    # FUNCTION TO FIND TIME FILES in self.PathL_timestamp
    def find_time_files(self):
        self.time_list = []
        for root, dirs, files in os.walk(self.PathL_rPPG):
            for name in files:
                if name.split('.')[0].endswith('_time'):
                    self.time_list.append(join(root,name))
        self.time_list = np.array(natsorted(self.time_list))
        
    # FLAG REFERING TO WETHER THE NUMBER OF GROUND TRUTH FILES, FACES FOLDERS, AND RPPG FILES ARE THE SAME
    def same_number_of_faces_GT_rPPG_time(self):
        return np.size(self.faces_list) == np.size(self.GT_list) == np.size(self.rPPG_list) == np.size(self.time_list)

    # FUNCTION TO SET ONE (OR MORE) SPECIFIC SUBJECTS
    def set_subject(self,PathL_Face,PathL_GT,PathL_rPPG):
        self.faces_list = PathL_Face
        self.GT_list = PathL_GT
        self.rPPG_list = PathL_rPPG
        
    # FUNCTION TO LOAD GT FILE + POSPROCESSING
    def load_GT(self,path):
        #Load Ground truth
        file_gt = []
        with open(path) as f:
            for index,row in enumerate(f):                
                if index > 0: # First row has the word "wave"                   
                    file_gt.append(float(row))

        return file_gt

    # FUNCTION TO LOAD TIMESTAMP FILE (IRREGULAR FRAME RATE IN VIPL)
    def load_time(self,path):
        time = np.loadtxt(path);       
        return time/1000

    # FUNCTION TO LOAD RPPG FILE + POSPROCESSING
    def load_rPPG(self,path):
        if self.dataset == 'MMSE':
            m = sio.loadmat(path)
            t = m['timeTrace'];t=np.resize(t,(t.shape[1],))
            pulseTrace = m['pulseTrace']; pulseTrace=np.resize(pulseTrace,(pulseTrace.shape[1],))#pltnow(pulseTrace,val=1,fr=25)
        if self.dataset == 'VIPL':
            # For VIPL we took PVM rPPG, .csv file
            pulseTrace = []
            with open(path) as f:
                for row in f:
                    pulseTrace.append(float(row.split(',')[0]))
            pulseTrace = np.array(pulseTrace)

        return pulseTrace
    
    # FUNCTION TO LOAD THE INDEX OF THE FRAMES OF ONE FACES FOLDER
    def load_faces_index(self,path):
        faces_idx = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith(('.png','.jpg')):
                    faces_idx.append(name)
        return np.array(natsorted(faces_idx))

    # FUNCTION TO LOAD THE INDEX OF THE FRAMES OF ONE FACES FOLDER
    def load_faces_index(self,path):
        faces_idx = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith(('.png','.jpg')):
                    faces_idx.append(name)
        return np.array(natsorted(faces_idx))

    # FUNCTION TO GET THE NEW NAMES OF FRAMES SYNCHRONIZED
    def SincronizameEsta(self,name,rPPG,GT,faces_idx,cuting,time):
        flag_Valid = True # Valid subject flag
        # Get rPPG fake frame rate from timestamp vector
        self.fr_rppg = round(time[-1]-time[0]/(len(time)))
        ## Resample GT to rPPG/frames timestamp
        gt_time = []
        for i in range(0,len(GT)):
            gt_time.append(i*1/self.fr_gt)
        # gt_time must have more or equal time than time
        while time[-1]>gt_time[-1]:
            # Sometimes we have to add values to get the equal or highest time in gt_time
            gt_time.append(gt_time[-1]+(1/self.fr_gt))# Add one more sample in time
            GT.append(GT[-1])# Add one more sample in value (I just re-use the last value)
        
        
        f = interp1d(gt_time,GT)
        GT_new = f(time)
        GT = GT_new.copy()
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
            time = time[0:GT.size]
        #pltnow(rPPG,GT,val=2,fr=25)

        # Take only the reliable segment
        self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),'Reliable segment in [{},{}] s.\n'.format(Begin_in_sec,End_in_sec))
        print('Reliable segment in [{},{}] s.'.format(Begin_in_sec,End_in_sec))
        cut_ini,_ = find_nearest(time,Begin_in_sec)#int(Begin_in_sec*self.fr_rppg)
        cut_end,_ = find_nearest(time,End_in_sec)#int(End_in_sec*self.fr_rppg)
        if cut_end <= 0: # If is 0 it means that End_in_sec was -1 (last position)
            GT = GT[cut_ini:]
            rPPG = rPPG[cut_ini:]
            faces_idx = faces_idx[cut_ini:]
            time = time[cut_ini:]
        else: # If End_in_sec was different than -1
            GT = GT[cut_ini:cut_end]
            rPPG = rPPG[cut_ini:cut_end]
            faces_idx = faces_idx[cut_ini:cut_end]
            time = time[cut_ini:cut_end]
        #pltnow(rPPG,GT,val=2,fr=25)

        # Start new time vector in 0
        time = time-time[0]
        
        # Take only signals with a length superior of MinLengthInSec
        if time[-1]<self.MinLengthInSec:#If signals is less than MinLengthInSec seconds:
            self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),'[ERROR]: size is {}, (less than {} seconds)\n'.format(time[-1],self.MinLengthInSec))
            print('[ERROR]: size is {}, (less than {} seconds)'.format(rPPG.size*1/self.fr_rppg,self.MinLengthInSec))
            faces_idx = []
            time = []
            flag_Valid = False
        else: # If the signal has a good size, do preprocessing in reliable segment
            # Synchronize rPPG in GT, the same shift must be aplied in faces_idx
            ROI_ini = 0*self.fr_rppg # begin in first 0 seconds
            ROI_end = ROI_ini+(5*self.fr_rppg)#finish 5 seconds latter
            # Center ground truth and rPPG in 0
            GT = GT-np.mean(GT)# pltnow(GT,val=1,fr=25)
            rPPG = rPPG-np.mean(rPPG)# pltnow(rPPG,val=1,fr=25)
            # Detrend signal
            GT = detrendsignal(GT) # pltnow(GT,val=1,fr=25)
            rPPG = detrendsignal(rPPG) # pltnow(rPPG,val=1,fr=25)
            # smooth the signal
            GT = smooth(GT, window_len=5, window='flat') # pltnow(GT,val=1,fr=25)
            rPPG = smooth(rPPG, window_len=5, window='flat') # pltnow(rPPG,val=1,fr=25)
            # Normalize between [-1,1]
            GT = normalize(GT);GT = np.resize(GT,GT.size) # pltnow(GT,val=1,fr=25)
            rPPG = normalize(rPPG);rPPG = np.resize(rPPG,rPPG.size) # pltnow(rPPG,val=1,fr=25)
            # GT alignment on rPPG
            #pltnow(GT[int(ROI_ini):int(ROI_end)],rPPG[int(ROI_ini):int(ROI_end)],2)
            s1 = phase_align(rPPG, GT, [ROI_ini,ROI_end])# align by phase
            GT = np.roll(GT, int(s1)) # pltnow(rPPG,GT,val=2,fr=25)
            self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),'Shifted {:0.2f} frames, ({:0.2f} seconds)\n'.format(s1,s1*1/self.fr_rppg))
            print('Shifted {:0.2f} frames, ({:0.2f} seconds)'.format(s1,s1*1/self.fr_rppg))

        return faces_idx,GT,time,flag_Valid #GT may changed

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
            name_subject_rppg = self.rPPG_list[i].split(os.path.sep)[-1].split('_rppg')[0]
            name_subject_time = self.time_list[i].split(os.path.sep)[-1].split('_time')[0]

            # face frames, GT file, and rPPG file must correspond to the same subject
            if name_subject_face == name_subject_GT == name_subject_rppg == name_subject_time:
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
                time = self.load_time(self.time_list[i])
                # For some reason the POS-rPPG signals has one less value so we duplicate the last one in order to have same length.
                # rPPG = np.concatenate((rPPG,(rPPG[-1],)),axis=0)
                # number of frames in "faces_idx" must be the same found in "rPPG" since rPPG was taken from the same video
                if len(rPPG)==len(faces_idx):
                    GT = self.load_GT(self.GT_list[i]) # Load GT signal ready to be compared. pltnow(GT,rPPG,val=3,fr=25)
                    frame_idx,newGT,newtime,valid = self.SincronizameEsta(name,rPPG,GT,faces_idx,cuting,time)
                    if valid: # If subject had a valid length, save it.
                        for j in range(0,len(frame_idx)):
                            current_frame = cv2.imread(join(self.PathL_Face,name,frame_idx[j]))
                            cv2.imwrite(join(self.PathS_Faces,name,frame_idx[j]), current_frame)
                        # Finally save new GT file because it may be changed, and also the timestamp file
                        self.saveArray_TXT(join(self.PathS_Faces,name,name+'_gt.txt'),[newGT])
                        self.saveArray_TXT(join(self.PathS_Faces,name,name+'_timestamp.txt'),[newtime])
                else:
                    self.report_TXT(join(self.PathS_Faces,'Dataset_Report.txt'),
                        "[ERROR]: rPPG length = {} and number of frames = {}. They should be the same\n".format(len(rPPG),len(faces_idx)))
                    print('{} has different length in rPPG and frames, skipping.'.format(name))
                    continue

            else:
                print('Files for Subject {} do not correspond. Face:{},GT:{},rppg:{}'.format(i,name_subject_face,name_subject_GT,name_subject_rppg))
                continue

#%% MAIN

VIPL = synchronize_faces(PathL_Face,PathL_GT,PathL_rPPG,PathS_Faces,MinimumSizeVideoInSeconds,'VIPL')
VIPL.find_faces_folders()
VIPL.find_GT_files()
VIPL.find_rPPG_files()
VIPL.find_time_files()
# Uncomment next lines to test one specific subject
# VIPL.set_subject([r'J:\faces\128_128\original\MMSE\F009_T11'],#faces
#                  [r'J:\faces\128_128\original\MMSE\F009_T11\F009_T11_gt.txt'],#GT
#                  [r'J:\POS_traces\MMSE-HR\filter\F009_T11_rppg_POS.mat'])#rPPG
if VIPL.same_number_of_faces_GT_rPPG_time():
    VIPL.TakeOnlyReliableSegmentAndSynchronize(r'E:\repos\face_cropper\source\removeGT\GT_SignalsToCut_VIPL.xlsx')
else:
    print('Error, different number of files in faces folders, GT files and/or rPPG files')

# CODE FOR REMOVE EMPTY FOLDERS IF NEEDED:
if 0:
    PathToRemoveEmptyFolders = PathS_Faces
    def is_dir_empty(path):
        with os.scandir(path) as scan:
            return next(scan, None) is None
    
    for root, dirs, files in os.walk(PathToRemoveEmptyFolders):
        for carpeta in dirs:
            if os.path.isdir(os.path.join(root,carpeta)) and is_dir_empty(os.path.join(root,carpeta)):
                os.rmdir(os.path.join(root,carpeta))