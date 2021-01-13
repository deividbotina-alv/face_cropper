'''
This cript takes the MMSE-HR dataset and cropp the faces, saving them with their
respective ground truth files.
'''
#%% GLOBAL VARIABLES
loadingPath = r'J:\Original_Datasets\MMSE-HR\MMSE-HR' # Path where we can find original files
savingPath = r'J:\faces\MMSE' # Path where faces will be saved
filespath = r'E:\repos\face_cropper\source' #Path with the files needed to run this script
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

class face_cropper_MMSE():
    def __init__(self,LoadPath,SavePath,filespath,SHOW=False):
        self.LoadPath = LoadPath
        self.SavePath = SavePath
        self.filespath = filespath
        self.SHOW = SHOW
        self.confidenceThreshold = 0.5
        self.critical_confidence = 0.99#if confidence is lower than critical_confidence, save the frame in text file (i.e. bad frame)

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

    # Flag Checking if there is the same number of files that the ground truth files
    def same_number_of_files(self):        
        return np.size(self.datapath) == np.size(self.gtpath)
    
    def report_TXT(self,nameFile,message):
        fileTXT = open(nameFile,"a")
        fileTXT.write(message)
        fileTXT.close()


    def detect_face(self, net, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,#image,scalefactor,size
                                     (300, 300), (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        detections = net.forward()
        
        best_detection = None
        maxConfidence = 0
        
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence < self.confidenceThreshold or confidence < maxConfidence:
                continue
            
            maxConfidence = confidence
            best_detection = detections[0, 0, i, 3:7]
            
            return (maxConfidence, best_detection)
        
    def get_frames_names(self,sujbect_path): #counf frames in one subject folder
            n_framesL = []
            for file in os.listdir(sujbect_path):
                if file.endswith(".jpg"):
                    n_framesL.append(join(sujbect_path,file))
            n_framesL = np.array(natsorted(n_framesL))
            return n_framesL

    # Return crop image according to the box
    def crop_image(self,frame, box):
        (x, y, w, h) = [int(v) for v in box]
        
        maxLength = max(h, w)
        x -= int((maxLength-w)/2)
        y -= int((maxLength-h)/2)
        h = maxLength
        w = maxLength

        crop_img = copy.deepcopy(frame[y:y+h, x:x+w])
        
        return crop_img

    def crop_NetFromCaffe(self,new_dimensions):
        print('Cropping faces by Network from caffe...')
        net = cv2.dnn.readNetFromCaffe(join(self.filespath,"deploy.prototxt.txt"),join(self.filespath,"res10_300x300_ssd_iter_140000.caffemodel"))
            # Start report of current dataset
        self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                        "[INFO] Face cropping report of dataset in " + self.LoadPath + '\n')
        # Crop faces subject by subject
        for i in range(0,len(self.datapath)):# Number of subjects
            unusual_subject = False #Flag refering to wether all faces were found and cropped properly in all frames
            has_to_detect_face = True
            initBB = None # initialize the bounding box coordinates of the object we are going to track
            subject = self.datapath[i].split(os.path.sep)[-2]+'_'+self.datapath[i].split(os.path.sep)[-1]
            if not os.path.exists(join(self.SavePath,subject)): os.makedirs(join(self.SavePath,subject))
            print('[INFO] Subject: '+ subject)
            # Start report in current subject
            self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                            "[INFO] This file corresponds to " + subject +
                            ".\nIt save problematic frames. If it is empty, each frame should be good.\n\n")
            # Count number of images in current subject
            n_framesL = self.get_frames_names(self.datapath[i])

            for j in range(0,len(n_framesL)):#Number of frames in this subject
                is_crop_img = False # Flag refering to 
                current_frame_jpg = n_framesL[j].split(os.path.sep)[-1]
                current_frame_png = current_frame_jpg.split('.')[0]+'.png'
                try: # Does the current frame exist in saving path?
                    frame = cv2.imread(join(self.SavePath,subject,current_frame_png))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Just to see if exist or not
                    continue
                except: #if it does not exist, crop the face and save it
                    try: # Try to find the face in current frame, crop, resize and save it
                        frame = cv2.imread(join(self.datapath[i],current_frame_jpg))
                        #Get size of frame
                        (H, W) = frame.shape[:2]
                        # 1) IF WE DON'T KNOW WHERE THE FACE IS: -> detect face
                        if has_to_detect_face:
                            print("[INFO] Launch face detection")
                            maxConfidence, best_detection = self.detect_face(net, frame)
    
                            if maxConfidence > 0:
                                # compute the (x, y)-coordinates of the bounding box for the object
                                box = best_detection * np.array([W, H, W, H])
    
                                (startX, startY, endX, endY) = box
                                
                                initBB = (startX, startY, endX-startX, endY-startY)
                                tracker = cv2.TrackerMedianFlow_create()
                                tracker.init(frame, initBB)
                                
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
                            
                            if success:
                                crop_img = self.crop_image(frame, box)
                                is_crop_img = True
                                
                                (crop_width, crop_height) = crop_img.shape[:2]
                                if crop_width > 0 and crop_height > 0:
                                    
                                    # If we don't detect face in crop_img, we should rerun dnn on the full frame
                                    maxConfidence, _ = self.detect_face(net, crop_img)
                                    
                                    if maxConfidence < self.critical_confidence:
                                        has_to_detect_face = True
                                        
                                (x, y, w, h) = [int(v) for v in box]
                                cv2.rectangle(frame, (x, y), (x+w, y+h),
                                              (0, 255, 0), 2)
                            else:
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
                                unusual_subject = True
                                print("[WARNING] Frame " + current_frame_jpg + " maxConfidence: "+ str(maxConfidence)+"\n")

                        else: # IF WE DIDN'T FIND A RELIABLE FACE, JUST SAVE AN EMPTY IMAGE TO PRESERVE THE SAME NUMBER OF FRAMES IN THE SUBJECT
                            empty = np.zeros(new_dimensions,dtype=float)
                            crop_img = cv2.merge((empty,empty,empty))
                            cv2.imwrite(join(self.SavePath,subject,current_frame_png), crop_img)
                            self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                                "[ERROR] Frame " + current_frame_jpg + "Error in cropping\n")
                            unusual_subject = True
                            print("[ERROR] Frame " + current_frame_jpg + "Error in cropping")
                    except:
                        # report something wrong happened
                        self.report_TXT(join(self.SavePath,subject,'Report.txt'),'ERROR in Frame '+current_frame_jpg.split('.')[0]+'\n')
                        unusual_subject = True
                        print('[ERROR]in Frame '+current_frame_jpg.split('.')[0])
                    if j == len(n_framesL)-1: # If last frame
                        if unusual_subject: #If either of the current subject's frames went wrong:
                             self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                                     "Subject " + subject + ': Unusual frames\n')
                        else:# If all current subject's frames were fine:
                             self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                                     "Subject " + subject + ': OK\n')
                             
    # Cropp images with OpenCv
    def crop_faces_face_cascadeV1(self,new_dimensions):
        print('Cropping faces by face_cascade...')
        rPPG = []
        bbox_list = []
        try: 
            faceCascade = cv2.CascadeClassifier(os.path.join(self.filespath,'haarcascade_frontalface_default.xml'))
        except:
            print('ERROR: '+os.path.join(self.filespath,'haarcascade_frontalface_default.xml')+' not found')
            sys.exit()

        # Crop faces subject by subject
        for i in range(0,len(self.datapath)):# Number of subjects
            subject = self.datapath[i].split(os.path.sep)[-2]+'_'+self.datapath[i].split(os.path.sep)[-1]
            if not os.path.exists(join(self.SavePath,subject)): os.makedirs(join(self.SavePath,subject))
            print('[INFO] Subject: '+ subject)
            
            # Count number of images in current subject
            n_framesL = []
            for file in os.listdir(self.datapath[i]):
                if file.endswith(".jpg"):
                    n_framesL.append(join(self.datapath[i],file))
            n_framesL = np.array(natsorted(n_framesL))

            for j in range(0,len(n_framesL)):#Number of images in this subject
                current_frame_jpg = n_framesL[j].split(os.path.sep)[-1]
                current_frame_png = current_frame_jpg.split('.')[0]+'.png'
                try: # Does the current frame exist in saving path?
                    img = cv2.imread(join(self.SavePath,subject,current_frame_png))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Just to see if exist or not
                    continue
                except: #if it does not exist, crop the face and save it
                    try: # Try to save face and save
                        img = cv2.imread(join(self.datapath[i],current_frame_jpg))
                        face_box = faceCascade.detectMultiScale(
                            img,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 30),
                            flags = cv2.CASCADE_SCALE_IMAGE
                        )
                        # Take only the first face founded
                        if len(face_box) > 1:
                            face_box = np.resize(face_box[0],(1,4)).copy()
        
                        # Crop face founded
                        (x, y, w, h) = (face_box[0,0],face_box[0,1],face_box[0,2],face_box[0,3])
        
                        maxLength = max(h, w)
                        x -= int((maxLength-w)/2)
                        y -= int((maxLength-h)/2)
                        h = maxLength
                        w = maxLength
    
                        crop_img = copy.deepcopy(img[y:y+h, x:x+w])
                        # Resize face
                        crop_img = cv2.resize(crop_img, new_dimensions, interpolation = cv2.INTER_AREA)
                        # Save current face
                        cv2.imwrite(join(self.SavePath,subject,current_frame_png), crop_img)
                        # report save img succesfully
                        self.report_TXT(join(self.SavePath,subject,'report.txt'),'Frame '+ current_frame_jpg.split('.')[0] +' OK\n')

                    except:
                        # report something wrong happened
                        self.report_TXT(join(self.SavePath,subject,'report.txt'),'ERROR in Frame '+current_frame_jpg.split('.')[0]+'\n')


MMSE = face_cropper_MMSE(loadingPath, savingPath, filespath, SHOW=False)
MMSE.find_files()
if not(MMSE.same_number_of_files()):
    print('Error: pathFiles and pathGT must have the same number of files')
    sys.exit()
else:
    #MMSE.crop_faces_face_cascadeV1((128,128))
    MMSE.crop_NetFromCaffe((128,128))