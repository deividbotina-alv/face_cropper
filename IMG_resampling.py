import numpy as np
from natsort import natsorted
from os.path import join
import os
import shutil as sh
import cv2

#%% MAIN
def main():
    ## 1. MMSE from 128_128 to 36_36
    # PathL = r'J:\faces\128_128\synchronized\MMSE'# Path to load dataset to be resampled
    # PathS = r'J:\faces\36_36\synchronized\MMSE'# Path to save resampled output
    # frame_file_type = 'png' # extension of frame files
    
    # MMSE = FrameResampler(PathL,PathS,frame_file_type)
    # MMSE.CopyPaste_files_And_ResampleImgs((36,36))
    
    ## 2. VIPL from 128_128 to 36_36
    PathL = r'J:\faces\128_128\original\VIPL'# Path to load dataset to be resampled
    PathS = r'J:\faces\36_36\original\VIPL'# Path to save resampled output
    frame_file_type = 'png' # extension of frame files
    
    VIPL = FrameResampler(PathL,PathS,frame_file_type)
    VIPL.CopyPaste_files_And_ResampleImgs((36,36))   
    


#%% CLASS MMSE_FrameResampler
class FrameResampler():
    def __init__(self,PathL,PathS,F_type):
        self.PathL = PathL
        self.PathS = PathS
        self.F_type = F_type
        self.List_of_Folders = [join(self.PathL,x) for x in os.listdir(self.PathL) if os.path.isdir(join(self.PathL,x))]
        self.List_of_Folders = np.array(natsorted(self.List_of_Folders))  
        print(f'=>{len(self.List_of_Folders)} Folder-Subjects found in {self.PathL}')

    def CopyPaste_files_And_ResampleImgs(self,new_dimensions):
        for i in range(0,len(self.List_of_Folders)):# For all subjects/folders
            subject = self.List_of_Folders[i].split(os.sep)[-1]
            print(f'=> Working in subject {subject}')
            # Create folder in Savepath
            if not os.path.exists(join(self.PathS,subject)):
                os.makedirs(join(self.PathS,subject))
            # Get all files in current subject/folder
            files = os.listdir(self.List_of_Folders[i])
            for j in range(0,len(files)):#For all files in current subject
                # If it is something else than png images then just copy and paste
                if files[j].split('.')[-1]!=self.F_type:
                    sh.copy(join(self.List_of_Folders[i],files[j]),join(self.PathS,subject,files[j]))
                else:
                    # If it is a img, then resample it
                    try: # Does the current frame exist in saving path?
                        frame = cv2.imread(join(self.PathS,subject,files[j]))
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Just to see if exist or not
                        continue # If exist skip to the next one
                    except: #if it does not exist, crop the face and save it
                        frame = cv2.imread(join(self.List_of_Folders[i],files[j]))
                        frame_resized = cv2.resize(frame, new_dimensions, interpolation = cv2.INTER_AREA)
                        cv2.imwrite(join(self.PathS,subject,files[j]), frame_resized)
                    
                
                    
                    
    # FUNCTION TO FIND FRAMES FILES in self.PathL_rPPG
    def find_frame_and_GT_files(self):
        self.frame_list = []
        self.GT_list = []
        for root, dirs, files in os.walk(self.PathL):
            for name in files:
                if name.split('.')[0].endswith(self.F_type):
                    self.frame_list.append(join(root,name))
                elif name.split('.')[0].endswith(self.GT_type):
                    self.GT_list.append(join(root,name))
                    
        self.frame_list = np.array(natsorted(self.rPPG_list))    
        self.GT_list = np.array(natsorted(self.GT_list))
    
    
        
        

if __name__ == '__main__':
    main()

