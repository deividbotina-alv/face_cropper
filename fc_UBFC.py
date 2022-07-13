'''
2022/07/11:
This cript takes the UBFC-rPPG dataset and cropp the faces, saving them with their
respective ground truth files.
Note: I am using the path J:\\Original_Datasets\\UBFC_DATASET\\DATASET_2 where data has been processed by Simon
for example, I have the pulseOx_gt.txt ground truth file and the folders vid.avi_faces with faces already cropped.
'''

#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from os import listdir
from os.path import join, abspath
import shutil as sh
from natsort import natsorted
import glob
import sys
import cv2
import copy
import mediapipe as mp
import argparse
import collections
from ubfc_functions import detrendsignal, normalize
#%% CLASSES AND FUNCTIONS

class FaceLandMarks():
    def __init__(self, staticMode=False,maxFace=1, refine_landmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace =  maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mp_drawing = mp.solutions.drawing_utils # To draw landmarks
        self.mp_face_mesh = mp.solutions.face_mesh # 
        self.faceMesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFace,
                                                 refine_landmarks=False,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon
                                                 )
        self.drawSpec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, version, draw=False):
        """
        version(int): Version 1, 2 or 3.
            if 1 Face is recropped based on landmarks borders
            if 2 Face is recropped based on landmarks borders + 5 pixels
            if 3 Face is not cropped. Only skin detection
        """
        #to be tuned for mask drawing, but 8% seems ok . I didnot tested in in MSE
        thickness_percent=8

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False # To improve performance, optionally mark the image as not writeable to pass by reference
        results = self.faceMesh.process(imgRGB)
        imgRGB.flags.writeable = True

        # Draw the pose annotation on the image.
        imageBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
        
        imgMask = np.zeros(imageBGR.shape, np.uint8)
        xmin=ymin=1
        xmax=ymax=0
   
        shape = imageBGR.shape
        image_rows, image_cols, _ = imageBGR.shape
        
        
        if results.multi_face_landmarks: # If face found and therefore landmarks
            
            face_landmarks=results.multi_face_landmarks[0]
            # GET RELATIVE MAXIMUM AND MINIMUM VALUES
            for landmark in face_landmarks.landmark:
    
                if (landmark.x<xmin):
                  xmin=landmark.x
                if (landmark.y<ymin):
                  ymin=landmark.y
                if (landmark.x>xmax):
                   xmax=landmark.x
                if (landmark.y>ymax):
                   ymax=landmark.y  
            
            relative_xmin = int(xmin * shape[1])
            if relative_xmin<0: relative_xmin=0
            relative_ymin = int(ymin * shape[0])
            if relative_ymin<0: relative_ymin=0
            relative_xmax = int(xmax * shape[1])
            if relative_xmax>image_cols+1: relative_xmax=image_cols+1
            relative_ymax = int(ymax * shape[0])
            if relative_ymax>image_rows+1: relative_ymax=image_rows+1 
            
            # Converte normalized face_landmarks to pixel indexes
            idx_to_coordinates = {}
            for idx, landmark in enumerate(face_landmarks.landmark):
              if ((landmark.HasField('visibility') and
                   landmark.visibility < mp._VISIBILITY_THRESHOLD) or
                  (landmark.HasField('presence') and
                   landmark.presence < mp._PRESENCE_THRESHOLD)):
                continue
              landmark_px = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                             image_cols, image_rows)
              if landmark_px:
                idx_to_coordinates[idx] = landmark_px
            
            # GET FACEMASK
            myconnections = self.mp_face_mesh.FACEMESH_FACE_OVAL
            num_landmarks = len(face_landmarks.landmark)
            k=0
            longueur=len(myconnections)
            points=np.zeros((longueur,2),np.int32)
            maski = np.zeros( (image_rows+2, image_cols+2), np.uint8)
            gx=0
            gy=0
            for connection in myconnections:
              start_idx = connection[0]
              end_idx = connection[1]
              if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
              if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                cv2.line(maski, idx_to_coordinates[start_idx],idx_to_coordinates[end_idx], (255,255,255),1)
                #cv2.imshow('masque', maski)
                gx=gx+idx_to_coordinates[start_idx][0]
                gy=gy+idx_to_coordinates[start_idx][1]
                if (k<longueur) : points[k]=idx_to_coordinates[start_idx]
                k=k+1
                
            if k!=0:
                gx = int (gx/k)
                gy = int (gy/k)
                cv2.floodFill(imgMask, maski, (gx,gy), (255,255,255));
                #cv2.imshow('masque', maski)
                #cv2.imshow('masque', imgMask)           

            # REMOVE EYES, EYEBROWS AND LIPS            
            if relative_xmin!=relative_xmax and relative_ymin!=relative_ymax:
            
                thick=(thickness_percent)*(relative_xmax-relative_xmin)/100;
                ith=int(thick)
                drawing_spec = self.mp_drawing.DrawingSpec(color = [0,0,0], thickness=ith, circle_radius=1)

                drawing_spec2 = self.mp_drawing.DrawingSpec(color = [255,255,255], thickness=ith, circle_radius=1)
                
                self.mp_drawing.draw_landmarks(
                    image=imgMask,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
                
                self.mp_drawing.draw_landmarks(
                    image=imgMask,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
                
                self.mp_drawing.draw_landmarks(
                    image=imgMask,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
                
                self.mp_drawing.draw_landmarks(
                    image=imgMask,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
                   
                self.mp_drawing.draw_landmarks(
                    image=imgMask,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
                
                if version==1:
                    # Recrop face following the landmarks borders
                    mask_crop=imgMask[relative_ymin:relative_ymax,relative_xmin:relative_xmax]
                    img_crop=imageBGR[relative_ymin:relative_ymax,relative_xmin:relative_xmax]
                elif version==2:
                    # Recrop face 5 pixels out of the landmarks borders
                    relative_xmin = relative_xmin - 5
                    if relative_xmin<0: relative_xmin=0
                    relative_xmax = relative_xmax + 5
                    if relative_xmax>image_cols+1: relative_xmax=image_cols+1
                    relative_ymin = relative_ymin - 5
                    if relative_ymin<0: relative_ymin=0
                    relative_ymax = relative_ymax + 5 
                    if relative_ymax>image_rows+1: relative_ymax=image_rows+1 
                    
                    mask_crop=imgMask[relative_ymin:relative_ymax,relative_xmin:relative_xmax]
                    img_crop=imageBGR[relative_ymin:relative_ymax,relative_xmin:relative_xmax]  
                elif version==3:
                    # Don't recrop.
                    mask_crop=imgMask
                    img_crop=imageBGR                    
                else:
                    print(f'version={version} is not a valid option')               
                if draw:
                    cv2.imshow('img_crop',img_crop)
                    cv2.imshow('mask_crop',mask_crop)

                return img_crop, mask_crop                      
        
        else: # If face is not detected then return same image and empty mask

            return imageBGR, np.zeros((imageBGR.shape[0],imageBGR.shape[1],3),dtype=np.uint8)


def SkinDetectionAndResizing(loadingPath:str,savingPath:str,newsize:int,saveskinmask:bool,color_channel:str,SHOW:bool=False):
    '''
    Function to take synchronized UBFC videos to:
        1) Detect face by landmarks
        2) Create boolean mask BoolSkinMask
        3) Resize input to newsize and also BoolSkinMask if needed
        4) Copy and paste remaining files without modifications
    '''
    # Start utils for face landmark detection and drawing
    DEBUG = False
    folders = natsorted([x[0] for x in os.walk(loadingPath)][1:])
    folders = [fold for fold in folders if fold.endswith('vid.avi_faces')]
    
    # Iterate all folders
    for folder in folders:
        # CREATE FOLDER PER SUBJECT
        subject = folder.split(os.path.sep)[-2]
        print(f'=>{subject}')
        if not(os.path.exists(join(savingPath,subject))): os.makedirs(join(savingPath,subject))  
        
        if len(os.listdir(folder)) == 0: 
            print('Empty folder')
        else: # If it is not empty then just do it !

            # DETECT SKIN, RESIZE AND SAVE FRAMES WITH MASKS
            try:
                framesORIG = np.load(join(savingPath,subject,subject+'.npy'))
                if saveskinmask:
                    BoolSkinMask = np.load(join(savingPath,subject,subject+'_skinmask.npy'))
            except:
                detector = FaceLandMarks()                
                frames_path = natsorted([file_path for file_path in os.listdir(folder) if file_path.endswith('.png')]) # load frames path
                framesRESIZED = np.zeros(shape=(len(frames_path),newsize,newsize,3),dtype=np.uint8)
                BoolSkinMask = np.zeros(shape=(len(frames_path),newsize,newsize),dtype=np.bool)
                for i in range(0,len(frames_path)):
                    if DEBUG: print(i)
                    frameBGR = cv2.imread(join(folder,frames_path[i]))#framesORIG[i,:,:,:]
                    # FIND FACE AND MASK
                    frame, mask = detector.findFaceLandmark(frameBGR,version=1,draw=False)
                    if color_channel == 'RGB':
                        frame = frame
                    elif color_channel == 'YUV':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    elif color_channel == 'YCrCb':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                    elif color_channel == 'HSV':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                        
                    elif color_channel == 'Lab':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab) 
                    elif color_channel == 'Luv':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Luv) 
                    #cv2.imshow('face',frame);cv2.imshow('mask',mask)
                    # RESIZE FACE AND MASK
                    frameRESIZED = cv2.resize(frame,(newsize,newsize), interpolation = cv2.INTER_AREA)
                    maskRESIZED = cv2.resize(mask,(newsize,newsize), interpolation = cv2.INTER_AREA)
                    #cv2.imshow('faceRESIZED',frameRESIZED);cv2.imshow('maskRESIZED',maskRESIZED)
                    #BoolmaskRESIZED = np.array(maskRESIZED[:,:,0],dtype=bool);plt.figure(),plt.imshow(BoolmaskRESIZED)
                    framesRESIZED[i,:,:,:] = frameRESIZED
                    BoolSkinMask[i,:,:] = np.array(maskRESIZED[:,:,0],dtype=bool)
                    
                    if SHOW:
                        img = framesRESIZED[i]
                        mask = BoolSkinMask[i]
                        img[~mask,:] = [0,0,0]
                        cv2.imshow('img',img)
                
                np.save(join(savingPath,subject,subject+'.npy'),framesRESIZED)
                if saveskinmask:
                    np.save(join(savingPath,subject,subject+'_skinmask.npy'),BoolSkinMask)
                # Release memory
                del detector; del frames_path; del framesRESIZED; del BoolSkinMask;del frameBGR; del frame; del frameRESIZED; del maskRESIZED 
                
            # COPY/PASTE GROUND TRUTH FILE
            try:
                sh.copy(join(folder.split('\\vid.avi_faces')[0],'pulseOx_gt.txt'),join(savingPath,subject,subject+'_gt.txt'))
                ground_truth = np.loadtxt(join(savingPath,subject,subject+'_gt.txt'))
                # Detrending
                ground_truth = detrendsignal(ground_truth)
                ground_truth = normalize(ground_truth)
                ground_truth = np.resize(ground_truth,(ground_truth.shape[0]))
                np.savetxt(join(savingPath,subject,subject+'_gt.txt'),ground_truth)
            except:
                print(f'ERROR saving {subject}_gt.txt')
            # COPY/PASTE TIMESTAMP FILE
            try:
                sh.copy(join(folder,subject+'_timestamp.txt'),join(savingPath,subject,subject+'_timestamp.txt'))
            except:
                print(f'ERROR saving {subject}_timestamp.txt')
                           
         
    print('end')

#%% MAIN
def main():
    print('================================================================')
    print('                FACE CROPPER UBFC-rPPG dataset                  ')
    print('================================================================') 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_skin_mask', action='store_true', default=False)
    parser.add_argument('-ch', '--color_channel', type=str, choices=['RGB','YUV','YCrCb','HSV','Lab','Luv'], default='RGB', required=True)  # 2022/02/27 Save Color different channels
    parser.add_argument('--newsize', type=int, choices=[128,64,32,16,8,4,2], default=8) # Only for sweep hyperparameter tuning
    parser.add_argument('-lp', '--load_path', type=str, required=True)
    parser.add_argument('-sp', '--save_path', type=str, required=True)    
    args = parser.parse_args()
    
    """""""""
    SHOW PARAMETERES CHOOSEN FOR THIS EXPERIMENT
    """""""""  
    for arg in vars(args):
        print(f'{arg} : {getattr(args, arg)}')
    print('================================================================')
    print('================================================================')     
    
    loadingPath = abspath(args.load_path) #r'J:\faces\128_128\synchronized\VIPL_npy\Facecascade'
    savingPath = abspath(args.save_path) #r'J:\faces\8_8\synchronized\VIPL_npy\MediapipeFromFascascade\HSV'
    SkinDetectionAndResizing(loadingPath,savingPath,args.newsize,args.save_skin_mask,args.color_channel,True)    

if __name__ == "__main__":
    main()  