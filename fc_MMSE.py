'''
This cript takes the MMSE-HR dataset and cropp the faces, saving them with their
respective ground truth files.
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
#%% CLASSES AND FUNCTIONS

class face_cropper_MMSE():
    # CONSTRUCTOR
    def __init__(self,LoadPath,SavePath,filespath,SHOW=False):
        self.LoadPath = LoadPath # Path with the dataset
        self.SavePath = SavePath # Path where faces will be saved
        self.filespath = filespath # Path with the files needed to run this script
        self.SHOW = SHOW # Show FACES croping
        self.confidenceThreshold = 0.5 
        self.critical_confidence = 0.99#if confidence is lower than critical_confidence, save the frame in text file (i.e. bad frame)
    
    # FUNCTION TO FIND PATHS WHERE DATA AND GT FILES ARE LOCATED
    def find_files(self):
        # CREATE FOLDER WHERE RESULTS WILL BE SAVED
        if not os.path.exists(self.SavePath): os.makedirs(self.SavePath)
        # Finding Folders where ground truth located
        folders_gt = [] 
        for root, dirs, files in os.walk(self.LoadPath):
            for i in files:
                if i.endswith("BP_mmHg.txt"):
                    folders_gt.append(root)
                    break

        # Finding Folders where the data is located
        folders_images = [] 
        for root, dirs, files in os.walk(self.LoadPath):
            for i in files:
                if i.endswith(".jpg"):         
                    folders_images.append(root)
                    break
        
        #Natural sort in files
        self.datapath = np.array(natsorted(folders_images))
        self.gtpath =  np.array(natsorted(folders_gt))
        
    # FUNCTION TO SET PATHS WHIT SUBJECTS
    def set_files(self, folders_images, folders_gt):
        #Natural sort in files
        self.datapath = np.array(natsorted(folders_images))
        self.gtpath =  np.array(natsorted(folders_gt))

    # Flag Checking if there is the same number of data files that the ground truth files
    def same_number_of_files(self):        
        return np.size(self.datapath) == np.size(self.gtpath)
    
    # FUNCTION TO REPORT SOMETHING IN A .TXT FILE
    def report_TXT(self,nameFile,message):
        fileTXT = open(nameFile,"a")
        fileTXT.write(message)
        fileTXT.close()
        
    # FUNCTION TO COPY THE ORIGINAL GROUND TRUTH FILES AND PASTE THEM IN SAVE PATH
    def copy_GT_files(self):
        # Dataset report
        print('[CG]Copy and pasting ground truth files...')
        self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                '\nCOPY AND PASTE GROUND TRUTH FILES\n\n')
        for path in self.gtpath:# Number of subjects 
            subject = path.split(os.path.sep)[-2]+'_'+path.split(os.path.sep)[-1] #subject name
            # Create folder for current subject if it does not exist
            if not os.path.exists(join(self.SavePath,subject)): os.makedirs(join(self.SavePath,subject))
            
            # Copy the respective ground truth with the same name but in .txt file
            try:
                sh.copy(join(path,'BP_mmHg.txt'),join(self.SavePath,subject,subject+'_gt.txt'))
                self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                    'Subject ' + subject + ': GT file OK\n')
            except:
                self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                    'Subject ' + subject + ': ERROR saving GT file\n')
        print('[CG]Process completed')
    # FUNCTION TO DETECT THE FACE IN A SPECIFIC FRAME: Usually used in first frame or whenever the tracker lost the face.
    def detect_face(self, frame):
        
        face_found = False
        #1) TRY TO FIND FACE BY Caffe network
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,#image,scalefactor,size
                                     (300, 300), (104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        best_detection = None
        maxConfidence = 0
        
        # IF THE NETWORK FOUND FACES
        if detections.shape[2]>0:

        # RETURN ONLY THE BEST DETECTION WITH THE MAXCONFIDENCE 
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]
        
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence < self.confidenceThreshold or confidence < maxConfidence:
                    continue
                
                maxConfidence = confidence
                best_detection = detections[0, 0, i, 3:7]
                
                face_found = True
        # IF CAFFE COULDN'T FIND ANY FACE
        else:
            face_found = False
            
        # #2) USE FACECASCADE INSTEAD -- NOT FINISHED NOR TESTED YET
        # if not(face_found):
        #     # We have to send the best_detection as the coffenetwork would do it, so we divide facecascade results by H and W
        #     (H, W) = frame.shape[:2]
        #     face_box = self.faceCascade.detectMultiScale(
        #         frame,
        #         scaleFactor=1.1,
        #         minNeighbors=5,
        #         minSize=(30, 30),
        #         flags = cv2.CASCADE_SCALE_IMAGE
        #     )
        #     # Take only the first face founded
        #     if len(face_box) < 1:
        #         maxConfidence = 0
        #         best_detection = None
        #     else: # If there is one or more faces                
        #         face_box = np.resize(face_box[0],(1,4)).copy()
        #         (x, y, w, h) = (face_box[0,0],face_box[0,1],face_box[0,2],face_box[0,3])
    
        #         maxLength = max(h, w)
        #         x -= int((maxLength-w)/2)
        #         y -= int((maxLength-h)/2)
        #         h = maxLength
        #         w = maxLength

            
        return (maxConfidence, best_detection)
        
    # FUNCTION TO EXTRACT THE FRAMES NAMES INSIDE THE CURRENT FOLDER
    def get_frames_names(self,sujbect_path): #counf frames in one subject folder
            n_framesL = []
            for file in os.listdir(sujbect_path):
                if file.endswith(".jpg"):
                    n_framesL.append(join(sujbect_path,file))
            n_framesL = np.array(natsorted(n_framesL))
            return n_framesL# List with the names per frame

    # FUNCTION TO RETURN CROP IMAGE ACCORDING TO THE BOS
    def crop_image(self,frame, box):
        (x, y, w, h) = [int(v) for v in box]
        
        maxLength = max(h, w)
        x -= int((maxLength-w)/2); 
        if x<0: x=0;
        y -= int((maxLength-h)/2);
        if y<0: y=0;
        h = maxLength
        w = maxLength

        crop_img = copy.deepcopy(frame[y:y+h, x:x+w])
        
        return crop_img

    def crop_faces(self,new_dimensions):
        
        print('[CF]Cropping faces by Network from caffe...')
        # LOAD CAFFE NETWORK FOR FACE DETECTION
        try:
            self.net = cv2.dnn.readNetFromCaffe(join(self.filespath,"deploy.prototxt.txt"),join(self.filespath,"res10_300x300_ssd_iter_140000.caffemodel"))
        except:
            print('ERROR: '+ join(self.filespath,'deploy.prototxt.txt or res10_300x300_ssd_iter_140000.caffemodel')+' not found')
            sys.exit()
        # LOAD FACECASCADE FOR FACE DETECTION
        try: 
            self.faceCascade = cv2.CascadeClassifier(os.path.join(self.filespath,'haarcascade_frontalface_default.xml'))
        except:
            print('ERROR: '+os.path.join(self.filespath,'haarcascade_frontalface_default.xml')+' not found')
            sys.exit()
        # Start report of current dataset
        self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                        "[INFO] Face cropping report of dataset in " + self.LoadPath + '\n')
        
        # Crop faces subject by subject
        for i in range(0,len(self.datapath)):# Number of subjects
            unusual_subject = False #Flag refering to wether all faces were found and cropped properly in all frames
            count_unusual = 0
            has_to_detect_face = True #Flag refering if is first frame or the tracker lost the face
            initBB = None # initialize the bounding box coordinates of the object we are going to track
            subject = self.datapath[i].split(os.path.sep)[-2]+'_'+self.datapath[i].split(os.path.sep)[-1] #subject name
            # Create folder for current subject
            if not os.path.exists(join(self.SavePath,subject)): os.makedirs(join(self.SavePath,subject))
            print('Subject: '+ subject)
            # Start report in current subject
            self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                            "[INFO] This file corresponds to " + subject +
                            ".\nIt save problematic frames. If it is empty, each frame should be good.\n\n")
            # Get names of all frames in current subject
            n_framesL = self.get_frames_names(self.datapath[i])

            for j in range(0,len(n_framesL)):#Number of frames in this subject
                is_crop_img = False # Flag refering to wheter the face was cropped succesfully from current frame
                crop_img = np.zeros((0,0,0))
                current_frame_jpg = n_framesL[j].split(os.path.sep)[-1] # subject_frame.jpg
                current_frame_png = current_frame_jpg.split('.')[0]+'.png' # subject_frame.png

                try: # Does the current frame exist in saving path?
                    frame = cv2.imread(join(self.SavePath,subject,current_frame_png))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Just to see if exist or not
                    continue # If exist skip to the next one
                except: #if it does not exist, crop the face and save it
                    try: # Try to find the face in current frame, crop, resize and save it
                        frame = cv2.imread(join(self.datapath[i],current_frame_jpg))
                        #Get size of frame
                        (H, W) = frame.shape[:2]
                        # 1) IF WE DON'T KNOW WHERE THE FACE IS: -> detect face
                        if has_to_detect_face:
                            print("[INFO] Launch face detection")
                            maxConfidence, best_detection = self.detect_face(frame)
    
                            if maxConfidence > 0:
                                # compute the (x, y)-coordinates of the bounding box for the object
                                box = best_detection * np.array([W, H, W, H])
    
                                (startX, startY, endX, endY) = box
                                # As we found the face, start tracking
                                initBB = (startX, startY, endX-startX, endY-startY)
                                tracker = cv2.TrackerMedianFlow_create()
                                tracker.init(frame, initBB)
                                # Crop the current face that we just found
                                crop_img = self.crop_image(frame, initBB)
                                is_crop_img = True
                                
                                (startX, startY, endX, endY) = box.astype("int")
                                
                                # draw the bounding box of the face along with the associated probability
                                text = "{:.2f}%".format(maxConfidence * 100)
                                y = startY - 10 if startY - 10 > 10 else startY + 10
                                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                              (0, 0, 255), 2)
                                cv2.putText(frame, text, (startX, y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                
                                has_to_detect_face = False
                                print("[INFO] Face found")

                        # 2) IF WE KNOW WHERE THE FACE IS, USE THE TRACKER
                        else: #hast_to_detect_face = False
                            (success, box) = tracker.update(frame)
                            
                            if success:# If it finds again the face
                                crop_img = self.crop_image(frame, box)
                                is_crop_img = True
                                
                                (crop_width, crop_height) = crop_img.shape[:2]
                                if crop_width > 0 and crop_height > 0:
                                    
                                    # If the face found by the tracker is not reliable, we should rerun facedetection in next frame
                                    maxConfidence, _ = self.detect_face(crop_img)                                    
                                    if maxConfidence < self.critical_confidence:
                                        has_to_detect_face = True
                                        
                                (x, y, w, h) = [int(v) for v in box]
                                cv2.rectangle(frame, (x, y), (x+w, y+h),
                                              (0, 255, 0), 2)
                            else: # if the tracker couldn't find the face
                                #Try to find the face without the tracker
                                print("[INFO] Launch face detection, tracker failed")
                                maxConfidence, best_detection = self.detect_face(frame)
                                if maxConfidence > 0:
                                    box = best_detection * np.array([W, H, W, H])    
                                    (startX, startY, endX, endY) = box
                                    initBB = (startX, startY, endX-startX, endY-startY)
                                    crop_img = self.crop_image(frame, initBB)
                                    is_crop_img = True
                                has_to_detect_face = True
                                
                        # SHOW FACE FOUND IN CURRENT FRAME IF NEEDED
                        if self.SHOW:
                            cv2.imshow("Frame", frame)
                        
                        # IF CROP FACE FOUND AND DIMMENSIONS ARE GOOD
                        if is_crop_img and crop_img.shape[0]>0 and crop_img.shape[1]>0:
                            if self.SHOW:
                                cv2.imshow("crop", crop_img)
                            # Resize face
                            crop_img = cv2.resize(crop_img, new_dimensions, interpolation = cv2.INTER_AREA)
                            cv2.imwrite(join(self.SavePath,subject,current_frame_png), crop_img)
                            if maxConfidence < self.critical_confidence:
                                self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                                                "[WARNING] Frame " + current_frame_jpg + " maxConfidence: "+ str(maxConfidence)+"\n")
                                unusual_subject = True; count_unusual += 1
                                print("[WARNING] Frame " + current_frame_jpg + " maxConfidence: "+ str(maxConfidence)+"\n")

                        else: # IF WE DIDN'T FIND A RELIABLE FACE, JUST SAVE AN EMPTY IMAGE TO PRESERVE THE SAME NUMBER OF FRAMES IN THE SUBJECT
                            empty = np.zeros(new_dimensions,dtype=float)
                            crop_img = cv2.merge((empty,empty,empty))
                            cv2.imwrite(join(self.SavePath,subject,current_frame_png), crop_img)
                            self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                                "[ERROR] Frame " + current_frame_jpg + " Error in cropping, saving ZEROS as result\n")
                            unusual_subject = True; count_unusual += 1
                            print("[ERROR] Frame " + current_frame_jpg + " Error in cropping, saving ZEROS as result")
                    except: #Excpetion of: Try to find the face in current frame, crop, resize and save it
                        # report something wrong happened
                        self.report_TXT(join(self.SavePath,subject,'Report.txt'),'ERROR in Frame '+current_frame_jpg.split('.')[0]+'\n')
                        unusual_subject = True; count_unusual += 1
                        print('[ERROR]in Frame '+current_frame_jpg.split('.')[0])
                    # REPORT IF ALL FRAMES OF CURRENT SUBJECT WERE OK OR NOT
                    if j == len(n_framesL)-1: # If last frame
                        if unusual_subject: #If either of the current subject's frames went wrong:
                             self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                                     "Subject " + subject + ': ' + str(count_unusual) + '/' + str(len(n_framesL))+'  Unusual frames\n')
                        else:# If all current subject's frames were fine:
                             self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                                             
                                     "Subject " + subject + ': OK\n')
        print('[CF]Process completed')

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
    Function to take 128x128 synchronized VIPL videos to:
        1) Detect face by landmarks
        2) Create boolean mask BoolSkinMask 128x128
        3) Resize input 128x128 to newsize and also BoolSkinMask if needed
        4) Copy and paste remaining files without modifications
    '''
    # Start utils for face landmark detection and drawing
    DEBUG = False
    folders = natsorted([x[0] for x in os.walk(loadingPath)][1:])
    # Iterate all folders
    for folder in folders:
        # CREATE FOLDER PER SUBJECT
        subject = folder.split(os.path.sep)[-1]
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
                framesnames = natsorted([x for x in os.listdir(folder) if x.endswith('.png')])
                framesRESIZED = np.zeros(shape=(len(framesnames),newsize,newsize,3),dtype=np.uint8)
                BoolSkinMask = np.zeros(shape=(len(framesnames),newsize,newsize),dtype=np.bool)
                for i in range(0,len(framesnames)):
                    if DEBUG: print(i)
                    frameBGR = cv2.imread(join(folder,framesnames[i]))
                    # FIND FACE AND MASK
                    frame, mask = detector.findFaceLandmark(frameBGR,version=1,draw=False)
                    # Just because in MMSE-HR we want to use the same 128x128 faces then ...
                    frame = frameBGR
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
                # del detector; del framesRESIZED; del BoolSkinMask;del frameBGR; del frame; del frameRESIZED; del maskRESIZED 
                
            # COPY/PASTE GROUND TRUTH FILE
            try:
                sh.copy(join(folder,subject+'_gt.txt'),join(savingPath,subject,subject+'_gt.txt'))
            except:
                print(f'ERROR savingg {subject}_gt.txt')
            # COPY/PASTE TIMESTAMP FILE
            try:
                sh.copy(join(folder,subject+'_timestamp.txt'),join(savingPath,subject,subject+'_timestamp.txt'))
            except:
                print(f'ERROR saving {subject}_timestamp.txt')
                           
         
    print('end')

#%% MAIN
def main():
    CROP_FACES_FROM_VIDEO = False
    SKIN_DETECTION_AND_RESIZE = True # 2021/10/29-> from faces already cropped

    #%% 2020/2021? I used face-cascade to crop faces, but then a synchronization was needed 
    if CROP_FACES_FROM_VIDEO:       

        loadingPath = r'J:\Original_Datasets\MMSE-HR\MMSE-HR' # Path where we can find original files
        savingPath = r'J:\faces\128_128\original\MMSE2' # Path where faces will be saved
        filespath = r'E:\repos\face_cropper' #Path with the files needed to run this script
        
        MMSE = face_cropper_MMSE(loadingPath, savingPath, filespath, SHOW=False)
        MMSE.find_files()
        # Uncomment next line to test one or multiple specific subjetcs
        # MMSE.set_files([r'J:\Original_Datasets\MMSE-HR\MMSE-HR\T10_T11_30Subjects\M014\T10',r'J:\Original_Datasets\MMSE-HR\MMSE-HR\first 10 subjects 2D\F013\T8'],
        #                 [r'J:\Original_Datasets\MMSE-HR\MMSE-HR\T10_T11_30PhyBPHRData\F010\T11',r'J:\Original_Datasets\MMSE-HR\MMSE-HR\first 10 subjects 2D\F013\T8'])
        if not(MMSE.same_number_of_files()):
            print('Error: pathFiles and pathGT must have the same number of files')
            sys.exit()
        else:
            MMSE.copy_GT_files()
            MMSE.crop_faces((128,128))        
    
    #%% 2022/07/20 I am copy-pasting the SKIN_DETECTION_AND_RESIZE from fc_VIPL. I need to take the the already
        #synchronized MMSE faces and generate the npy. files for cross dataset              
    elif SKIN_DETECTION_AND_RESIZE:        
        print('================================================================')
        print('              Running Skin Detection And Resizing               ')
        print('================================================================') 
        print('2022/07/20: Skin detection has no sence because we are just     ')
        print('Using the original 128x128 images, but the mask is based on face')
        print('detection                                                       ')
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_skin_mask', action='store_true', default=False)
        parser.add_argument('-ch', '--color_channel', type=str, choices=['RGB','YUV','YCrCb','HSV','Lab','Luv'], default='RGB', required=True)  # 2022/02/27 Save Color different channels
        parser.add_argument('--newsize', type=int, choices=[128,64,32,16,8,4,2], default=8) # Only for sweep hyperparameter tuning
        parser.add_argument('-lp', '--load_path', type=str, required=True)
        parser.add_argument('-sp', '--save_path', type=str, required=True)    
        args = parser.parse_args()
        
        for arg in vars(args):
            print(f'{arg} : {getattr(args, arg)}')
        print('================================================================')
        print('================================================================') 
        
        print(f'Saving skin masks (more space in disk needed): {args.save_skin_mask}')
        print(f'Saving {args.color_channel} channels')
        
        loadingPath = abspath(args.load_path) #r'J:\faces\128_128\synchronized\VIPL_npy\Facecascade'
        savingPath = abspath(args.save_path) #r'J:\faces\8_8\synchronized\VIPL_npy\MediapipeFromFascascade\HSV'
        SkinDetectionAndResizing(loadingPath,savingPath,args.newsize,args.save_skin_mask,args.color_channel,False)

if __name__ == "__main__":
    main()  